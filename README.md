# 🧵 Cloth Defect Detection
Fabric defect detection using **Swin Transformer + FPN + Faster R-CNN**, trained on the [ZJU-Leapers dataset](https://www.kaggle.com/datasets/muhammadhamzanawaz/cloths-defect-detection) (12,100+ images, 12 defect classes). Detects and localizes fabric defects with bounding boxes and per-class confidence scores.

---

## 🏷️ Classes
`Blue Plaid` · `Brown Plaid` · `Dot Pattern` · `Gingham` · `Gray Plaid` · `Houndstooth` · `Knot Pattern` · `Red Plaid` · `Thick Stripe` · `Thin Stripe` · `Twill Plaid` · `White Plain`

---

## 🗂️ Project Structure
```
cloth-defect-detection/
├── detect.py               # Local inference script
├── best_swin_cascade.pth   # Model checkpoint (download separately — see below)
└── result.png              # Auto-saved output after each detection
```

---

## ⚙️ Installation

```bash
pip install torch torchvision timm Pillow matplotlib kagglehub
```

---

## 📥 Download the Model

The trained checkpoint is hosted on Kaggle. Download it with:

```python
import kagglehub

path = kagglehub.model_download("abdullahsufian/swin-cascade/pyTorch/default")
print("Path to model files:", path)
```

Then copy `best_swin_cascade.pth` into the same folder as `detect.py`.

---

## 🚀 Usage

```bash
python detect.py
```

1. A **file picker dialog** opens — select any fabric image (`.jpg`, `.png`, `.bmp`)
2. The model runs inference and prints results to the terminal
3. A window opens showing the **original vs annotated image** with bounding boxes
4. The annotated result is saved as `result.png` in the project folder
5. You are asked if you want to detect another image

> **No GUI?** If running in a headless environment, the script falls back to terminal path input.

### Example Output

```
────────────────────────────────────────────────────────────
  Image   : fabric_sample.jpg
  Size    : 1024 × 768 px
  Device  : CUDA
────────────────────────────────────────────────────────────
  ⚠️   DEFECT DETECTED — 2 region(s) found

  #    Class                  Confidence   Bounding Box [x1, y1, x2, y2]
  ──────────────────────────────────────────────────────────────────────
  1    Houndstooth                 0.871   [112, 204, 389, 461]
  2    Gray Plaid                  0.743   [530, 180, 812, 420]
────────────────────────────────────────────────────────────

  💾 Result saved to: /your/path/result.png
```

---

## 🏗️ Model Architecture

| Component     | Details                                      |
|---------------|----------------------------------------------|
| Backbone      | Swin-Tiny (patch4, window7)                  |
| Neck          | Feature Pyramid Network (FPN) — 256 channels |
| Head          | Faster R-CNN with MultiScale RoI Align       |
| Input size    | 512 × 512                                    |
| Classes       | 12 defect classes + background               |
| Trained on    | Kaggle T4 GPU                                |

---

## 📊 Dataset

- **Name:** ZJU-Leapers Cloth Defect Detection
- **Images:** 12,100+
- **Classes:** 12
- **Source:** [Kaggle](https://www.kaggle.com/datasets/muhammadhamzanawaz/cloths-defect-detection)
- **Format:** YOLO-style labels (converted to bounding boxes for Faster R-CNN)

---

## 🔧 Configuration

You can tweak the following in `detect.py`:

| Parameter     | Default | Description                          |
|---------------|---------|--------------------------------------|
| `img_size`    | 512     | Input resolution                     |
| `conf_thresh` | 0.4     | Minimum confidence to show detection |

---

## 📦 Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- timm
- Pillow
- matplotlib
- kagglehub
