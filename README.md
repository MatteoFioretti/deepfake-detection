# Deepfake Detection via Transfer Learning on Face Crops

Binary classification of manipulated faces using ResNet-50 fine-tuned on FaceForensics++ C23, with a controlled ablation study quantifying the value of ImageNet pretraining.

---

## Author

| Name | Student ID | Email |
|------|-----------|-------|
| Matteo Fioretti | 2022346 | fioretti.2022346@studenti.uniroma1.it |

**Course:** AI Lab: Computer Vision – Signal Processing – NLP  
**Degree:** Bachelor's in ACSAI – Sapienza University of Rome – A.Y. 2025/2026  
**Repository:** [github.com/MatteoFioretti/deepfake-detection](https://github.com/MatteoFioretti/deepfake-detection)

---

## Project Overview

This project trains a binary deepfake detector on face crops extracted from the FaceForensics++ C23 dataset. The core contribution is a **controlled ablation study** comparing:

- **Fine-tuned model** — ResNet-50 initialized with ImageNet pretrained weights
- **From-scratch baseline** — identical architecture with random weight initialization

The goal is to isolate and quantify the value of transfer learning for forensic face analysis.

---

## Results

| Metric | Fine-Tuned | From Scratch |
|--------|-----------|--------------|
| Accuracy | 91.1% | 68.7% |
| Precision | 94.7% | 83.7% |
| Recall | 93.3% | 71.7% |
| F1 Score | 94.0% | 77.2% |
| ROC AUC | 0.960 | 0.739 |

---

## Repository Structure

```
deepfake-detection/
├── preprocessing.ipynb       # Face extraction pipeline (MTCNN, video-level split)
├── training.ipynb            # Model training (fine-tuned + from-scratch)
├── ablation.ipynb            # Evaluation metrics, confusion matrices, training curves
└── README.md
```

---

## Dataset

**FaceForensics++ C23** — 1,000 original videos manipulated with 5 methods:
- Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter

**Raw dataset (Kaggle):**  
[kaggle.com/datasets/xdxd003/ff-c23](https://www.kaggle.com/datasets/xdxd003/ff-c23)

**Preprocessed face crops (Kaggle):**  
[kaggle.com/datasets/matteofioretti/deepfake-processed](https://www.kaggle.com/datasets/matteofioretti/deepfake-processed)  
38,723 face crops organized as `processed/{split}/{class}/` — ready for `ImageFolder`.

---

## Model Weights

Trained checkpoints are stored on Kaggle (decoupled from notebook versions):  
[kaggle.com/datasets/matteofioretti/deepfake-checkpoints](https://www.kaggle.com/datasets/matteofioretti/deepfake-checkpoints)

| File | Description |
|------|-------------|
| `best_model_finetuned.pth` | ResNet-50 fine-tuned, best val loss checkpoint |
| `best_model_scratch.pth` | ResNet-50 from scratch, best val loss checkpoint |

To load a checkpoint:
```python
import torch
import torchvision.models as models

model = models.resnet50()
model.fc = torch.nn.Linear(2048, 2)
model.load_state_dict(torch.load("best_model_finetuned.pth"))
model.eval()
```

---

## Environment

| Dependency | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | latest stable |
| torchvision | latest stable |
| facenet-pytorch | 2.5.2 |
| OpenCV (cv2) | latest stable |
| scikit-learn | latest stable |
| pandas | latest stable |
| NumPy | latest stable |
| Pillow | latest stable |

All notebooks are designed to run on **Kaggle T4 GPU**. Set `KAGGLE = True` at the top of each notebook when running on Kaggle, `KAGGLE = False` for local execution.

---

## How to Reproduce

### Step 1 — Preprocessing

Run `preprocessing.ipynb` on Kaggle with the raw FF++ C23 dataset attached as input.

This notebook:
- Splits 1,000 original videos into train/val/test (80/10/10) at **video level** to prevent data leakage
- Applies the same split to each of the 5 manipulation method folders independently (per-manipulation balanced sampling)
- Extracts 10 uniformly sampled frames per video using OpenCV
- Detects and crops faces using MTCNN (`facenet-pytorch`, `keep_all=False`)
- Saves 224×224 face crops as JPEG organized by split and class

Output: the preprocessed dataset uploaded to the `deepfake-processed` Kaggle dataset.

### Step 2 — Training

Run `training.ipynb` on Kaggle with the preprocessed dataset attached as input.

This notebook:
- Loads face crops using `ImageFolder` with ImageNet normalization
- Applies training augmentations: `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `ColorJitter` (brightness/contrast only — no hue/saturation to preserve forensic color signals)
- Fine-tunes ResNet-50 (ImageNet pretrained) with differential learning rates: `1e-4` backbone, `1e-3` head
- Trains an identical from-scratch baseline with `weights=None`
- Uses weighted `CrossEntropyLoss` to handle 3:1 fake/real class imbalance
- Saves best checkpoint (minimum val loss) to the `deepfake-checkpoints` Kaggle dataset

### Step 3 — Evaluation

Run `ablation.ipynb` on Kaggle with the preprocessed dataset and model checkpoints attached as input.

This notebook:
- Loads both checkpoints and evaluates on the held-out test set
- Reports Accuracy, Precision, Recall, F1, ROC AUC for both models
- Plots confusion matrices and training curves for the ablation comparison

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Video-level split | Prevents data leakage — frames from the same video are visually near-identical |
| Per-manipulation balanced sampling | Ensures all 5 manipulation types are proportionally represented in every split |
| MTCNN over Haar cascades | Better accuracy on partial occlusion and varied pose; tighter crops preserve forensic boundary artifacts |
| 10 frames/video | Balances temporal coverage against compute budget |
| No hue/saturation jitter | Color artifacts are genuine forensic signals — augmenting them destroys discriminative information |
| ImageNet normalization | Mandatory for pretrained weights; filters were optimized for this input distribution |
| Weighted CrossEntropyLoss | Corrects 3:1 fake/real imbalance; formula: w_c = N / (2 · N_c) |
| Differential LR (1e-4 / 1e-3) | Conservative LR for pretrained backbone to avoid catastrophic forgetting; larger LR for randomly initialized head |
| Checkpoint on min val loss | Smoother and more sensitive signal than accuracy for selecting the best model state |

---

## Limitations

- **No temporal modeling** — frames classified independently; temporal artifacts (flickering, inconsistent blinking) are ignored
- **Dataset age** — FF++ predates diffusion-based and NeRF-based synthesis; model may not generalize to modern methods
- **Pixel space only** — no frequency-domain analysis; GAN high-frequency artifacts (Qian et al., 2020) are not explicitly targeted
- **MTCNN silent failure** — frames with no detected face are discarded without logging; failure rate may correlate with manipulation type
- **Frame-level evaluation** — metrics reported per frame, not per video; video-level aggregation would be more realistic
