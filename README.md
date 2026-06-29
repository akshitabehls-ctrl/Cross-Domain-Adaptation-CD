
# Cross-Domain Change Detection using Unsupervised Domain Adaptation

<div align="center">

### A PyTorch framework for robust cross-domain remote sensing change detection

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Task](https://img.shields.io/badge/Task-Remote%20Sensing%20Change%20Detection-success)
![Domain](https://img.shields.io/badge/Domain-Unsupervised%20Domain%20Adaptation-orange)
![Status](https://img.shields.io/badge/Status-Research-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

> **Official PyTorch implementation** of a framework for **Cross-Domain Change Detection** using **Unsupervised Domain Adaptation (UDA)**. The framework combines a Siamese ResNet-50 encoder, reconstruction-guided learning, adversarial domain adaptation, Sinkhorn-based pseudo-label refinement, and consistency regularization to improve change detection across heterogeneous remote sensing datasets.

<p align="center">
<img src="assets/framework.png" width="900">
</p>

> **Architecture Overview** *(Replace with your pipeline figure from the report.)*

---

# 📖 Overview

Remote sensing change detection aims to identify meaningful structural and semantic changes between images of the same geographical region captured at different times.

Although modern deep learning methods perform well on benchmark datasets, they often fail to generalize to unseen domains because of:

- Different satellite sensors
- Geographic variation
- Seasonal changes
- Illumination differences
- Resolution mismatch
- Environmental variations

This project addresses these challenges through **Unsupervised Domain Adaptation**, allowing a model trained on a labeled **source domain** to adapt effectively to an **unlabeled target domain**.

---

# ✨ Key Features

| Feature | Description |
|---------|-------------|
| Siamese ResNet-50 Encoder | Shared-weight feature extraction |
| Multi-scale Decoder | Pixel-level change prediction |
| Reconstruction Branch | Learns domain-invariant representations |
| Domain Discriminator | Gradient Reversal Layer (GRL) based alignment |
| Sinkhorn-Knopp | Balanced pseudo-label refinement |
| AWDA | Adaptive Weight Domain Adaptation |
| Consistency Regularization | Weak–Strong augmentation training |
| Framework | PyTorch |

---

# 📰 Updates

- Initial implementation released.
- Added reconstruction-guided training.
- Added Sinkhorn-based pseudo-label refinement.
- Added adversarial domain adaptation.
- Added consistency regularization.
- Evaluated on **LEVIR-CD → WHU-CD**.

---

# 🗂 Repository Structure

```text
Cross-Domain-Adaptation-CD
│
├── models
│   ├── clip_encoder.py
│   ├── decoder.py
│   └── discriminator.py
│
├── utils
│   ├── awda_loss.py
│   ├── focal_loss.py
│   ├── metrics.py
│   └── sinkhorn.py
│
├── initial_train.py
├── reconstruct.py
├── reconstruction_test.py
├── main_seperate.py
├── test.py
├── test_seperate.py
├── README.md
└── docs
    └── Project_Report.pdf
```

---

# 🏛 Methodology

The proposed framework consists of:

1. Siamese ResNet-50 Encoder
2. Multi-scale Decoder
3. Reconstruction Learning
4. Domain Adversarial Learning
5. Sinkhorn-based Pseudo-label Refinement
6. Adaptive Weight Domain Adaptation
7. Consistency Regularization

Training pipeline:

```text
Source Images
      │
      ▼
Siamese Feature Extraction
      │
      ▼
Reconstruction Learning
      │
      ▼
Pseudo Label Generation
      │
      ▼
Sinkhorn Refinement
      │
      ▼
Domain Adaptation
      │
      ▼
Consistency Training
      │
      ▼
Prediction
```

---

# 📂 Datasets

This project follows a **cross-domain setting**:

| Source Domain | Target Domain |
|--------------|---------------|
| LEVIR-CD | WHU-CD |

Expected structure:

```text
datasets/
├── LEVIR-CD
│   ├── train
│   ├── val
│   └── test
└── WHU-CD
    ├── train
    ├── val
    └── test
```

---

# ⚙ Installation

```bash
git clone https://github.com/akshitabehls-ctrl/Cross-Domain-Adaptation-CD.git
cd Cross-Domain-Adaptation-CD

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

---

# 🚀 Training

Initial reconstruction training

```bash
python initial_train.py
```

Domain adaptation training

```bash
python main_seperate.py
```

---

# 🧪 Testing

```bash
python test.py
```

Reconstruction evaluation

```bash
python reconstruction_test.py
```

---

# 📊 Experimental Results

## Quantitative Results

| Model | Precision | Recall | F1 | IoU |
|------|----------:|-------:|------:|------:|
| AWDA (Reference) | 85.34 | 80.51 | 82.85 | 70.73 |
| Proposed Framework | 82.64 | 58.35 | 68.41 | 51.98 |

---

## Qualitative Results

<p align="center">
<img src="assets/results.png" width="900">
</p>

---

## Ablation Study

| Configuration | Precision | Recall | F1 | IoU |
|--------------|----------:|-------:|------:|------:|
| Full Model | 82.64 | 58.35 | 68.41 | 51.98 |
| Without λ_adv | 67.07 | 38.29 | 48.75 | 32.23 |
| Without CWST | 52.09 | 39.31 | 44.80 | 28.87 |
| Without L_sup | 56.08 | 46.70 | 50.96 | 34.19 |

---

# 📈 Future Work

- Vision Transformer based encoders
- CLIP-based semantic representations
- Test-Time Adaptation
- Multi-source domain adaptation
- SAR + Optical fusion
- Multi-class change detection
- Better pseudo-label confidence estimation

---

# 📚 Citation

```bibtex
@misc{behl2026crossdomaincd,
  title={Cross-Domain Change Detection using Unsupervised Domain Adaptation},
  author={Akshita Behl and Shyamsundar Paramasivam},
  year={2026},
  note={Bachelor's Thesis, LNMIIT}
}
```

---

# 🙏 Acknowledgements

This project was inspired by recent work in:

- AWDA
- DANN
- U-Net
- LEVIR-CD
- WHU-CD
- PyTorch

Special thanks to **Dr. Ankit Jha** for guidance throughout the project.

---

# 👩‍💻 Authors

**Akshita Behl**  
B.Tech Computer Science Engineering  
The LNM Institute of Information Technology

**Shyamsundar Paramasivam**

Supervisor: **Dr. Ankit Jha**

---

# ⭐ If you found this repository useful

Please consider giving the repository a ⭐.

It helps others discover the project and motivates future research.
