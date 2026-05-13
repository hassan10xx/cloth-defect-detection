"""
Fabric Defect Detection — Local Inference
==========================================
Requirements:
    pip install torch torchvision timm torchmetrics Pillow matplotlib

Usage:
    python detect.py
    → A file dialog will open. Select any fabric image (.jpg / .png).
    → The script prints detected defects and opens an annotated result window.

Make sure 'best_swin_cascade.pth' is in the same folder as this script.
"""

import os
import sys
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
import torchvision.transforms.functional as TF
import timm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "nc"         : 12,
    "names"      : ["Blue Plaid", "Brown Plaid", "Dot Pattern", "Gingham",
                    "Gray Plaid", "Houndstooth", "Knot Pattern", "Red Plaid",
                    "Thick Stripe", "Thin Stripe", "Twill Plaid", "White Plain"],
    "img_size"   : 512,
    "conf_thresh": 0.4,
    "device"     : "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt"       : os.path.join(SCRIPT_DIR, "best_swin_cascade.pth"),
}

DEVICE      = torch.device(CFG["device"])
NUM_CLASSES = CFG["nc"] + 1
COLORS      = plt.cm.hsv(np.linspace(0, 1, CFG["nc"] + 1))


# ── Model ─────────────────────────────────────────────────────
class SwinBackboneWithFPN(torch.nn.Module):
    def __init__(self, img_size=512):
        super().__init__()
        self.body = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False,
            features_only=True, out_indices=(1, 2, 3),
            img_size=img_size, strict_img_size=False,
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[192, 384, 768],
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 256

    def forward(self, x):
        feats = {}
        for i, f in enumerate(self.body(x)):
            if f.ndim == 4 and f.shape[-1] != f.shape[1]:
                f = f.permute(0, 3, 1, 2).contiguous()
            feats[str(i)] = f
        return self.fpn(feats)


def build_model(num_classes, img_size=512):
    backbone = SwinBackboneWithFPN(img_size)
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        ),
        box_roi_pool=MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2),
        rpn_pre_nms_top_n_train=3000, rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1500,  rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7, box_score_thresh=0.05, box_nms_thresh=0.5,
        box_detections_per_img=200, min_size=img_size, max_size=img_size,
    )
    return model


# ── Load checkpoint ───────────────────────────────────────────
def load_model():
    if not os.path.exists(CFG["ckpt"]):
        print(f"\n❌  Checkpoint not found: {CFG['ckpt']}")
        print("    Make sure 'best_swin_cascade.pth' is in the same folder as detect.py")
        sys.exit(1)

    print(f"Loading model from: {CFG['ckpt']}")
    model = build_model(NUM_CLASSES, CFG["img_size"])
    ckpt  = torch.load(CFG["ckpt"], map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    print(f"✅ Model loaded | epoch={ckpt.get('epoch', '?')} | mAP@0.5={ckpt.get('map50', 0):.4f}")
    return model


# ── Pick image ────────────────────────────────────────────────
def pick_image():
    """Open a file dialog to select an image. Falls back to terminal input."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a fabric image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        root.destroy()
        return path if path else None
    except Exception:
        # Fallback for headless environments
        path = input("\nEnter full path to image file: ").strip().strip('"').strip("'")
        return path if os.path.exists(path) else None


# ── Run detection ─────────────────────────────────────────────
@torch.no_grad()
def detect(model, img_path):
    img_pil  = Image.open(img_path).convert("RGB")
    orig_W, orig_H = img_pil.size

    img_resized = img_pil.resize((CFG["img_size"], CFG["img_size"]))
    img_tensor  = TF.to_tensor(img_resized).unsqueeze(0).to(DEVICE)

    output = model(img_tensor)[0]

    keep   = output["scores"] >= CFG["conf_thresh"]
    boxes  = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()

    # Scale boxes back to original resolution
    sx = orig_W / CFG["img_size"]
    sy = orig_H / CFG["img_size"]
    if len(boxes):
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

    return img_pil, boxes, labels, scores


# ── Show result ───────────────────────────────────────────────
def show_result(img_pil, boxes, labels, scores, img_path):
    n = len(boxes)
    fname = os.path.basename(img_path)

    print(f"\n{'─'*60}")
    print(f"  Image   : {fname}")
    print(f"  Size    : {img_pil.width} × {img_pil.height} px")
    print(f"  Device  : {CFG['device'].upper()}")
    print(f"{'─'*60}")

    if n == 0:
        print("  ✅  NO DEFECT DETECTED")
        print(f"     (no prediction above confidence threshold {CFG['conf_thresh']})")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(img_pil)
        ax.set_title("✅  No Defect Detected", fontsize=13, color="green", fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        return

    print(f"  ⚠️   DEFECT DETECTED — {n} region(s) found\n")
    print(f"  {'#':<4} {'Class':<22} {'Confidence':>12}   {'Bounding Box [x1, y1, x2, y2]'}")
    print(f"  {'─'*70}")
    for i, (box, lbl, sc) in enumerate(zip(boxes, labels, scores)):
        name = CFG["names"][int(lbl) - 1] if 0 < int(lbl) <= CFG["nc"] else "unknown"
        print(f"  {i+1:<4} {name:<22} {sc:>11.3f}   "
              f"[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
    print(f"{'─'*60}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img_pil)
    for box, lbl, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        c    = COLORS[int(lbl) % len(COLORS)]
        name = CFG["names"][int(lbl) - 1] if 0 < int(lbl) <= CFG["nc"] else "unknown"
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=c, facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, max(y1 - 6, 0),
            f"{name}  {sc:.2f}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor=c, alpha=0.75, pad=2, edgecolor="none")
        )

    axes[1].set_title(f"⚠️  {n} Defect(s) Detected", fontsize=11,
                      color="red", fontweight="bold")
    axes[1].axis("off")

    plt.suptitle("Swin-T + FPN + Faster R-CNN — Fabric Defect Detection",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(SCRIPT_DIR, "result.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  💾 Result saved to: {save_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🧵  Fabric Defect Inspector — Swin-T + FPN + Faster R-CNN")
    print("=" * 60)

    model = load_model()

    while True:
        print("\nOpening file picker… (close dialog to exit)")
        img_path = pick_image()

        if not img_path:
            print("No image selected. Exiting.")
            break

        if not os.path.exists(img_path):
            print(f"❌  File not found: {img_path}")
            continue

        img_pil, boxes, labels, scores = detect(model, img_path)
        show_result(img_pil, boxes, labels, scores, img_path)

        again = input("\nDetect another image? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break
