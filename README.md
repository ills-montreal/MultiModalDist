# 🌟 Vision Distillation

📦 This repository contains the implementation of our distillation method for vision modalities. The objective is to distill knowledge from multiple teacher embedders into a single student model.

## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.10.14 🐍
- PyTorch 🔥
- torchvision 🖼️

### 📂 Dataset Preparation
Before training, you need to extract the embeddings of the dataset using the desired teacher models. The extracted embeddings must be saved in a specified directory.

**Available Datasets:**
- 📸 CIFAR10
- 🌐 DTD
- 🛩️ STL10
- 🚦 SVHN
- ✈️ FGVCAircraft
- 🐦 CUB

**Available Teacher Models:**
- 🌀 Swin
- 🔍 ViT
- 📖 BEiT
- 🦾 DINOv2
- 🚀 PVTv2

### 🏋️‍♂️ Training
After extracting embeddings, run the training script:

```bash
python multiDist/train_gm.py --modalities-to-simulate vision \
                             --embeddings-dir [EMBEDDINGS_PATH] \
                             --vision-student [STUDENT_MODEL] \
                             --vision-embedders-to-simulate [MODEL_NAME]
```

Example:

```bash
python multiDist/train_gm.py --modalities-to-simulate vision \
                             --embeddings-dir ./embeddings/ \
                             --vision-student Swin \
                             --vision-embedders-to-simulate ViT DINOv2
```

### 🛠️ Model Options
- **Student Models:** 🌀 Swin, 🔍 ViT, 📖 BEiT, 🦾 DINOv2, 🚀 PVTv2
- **Teacher Models:** 🌀 Swin, 🔍 ViT, 📖 BEiT, 🦾 DINOv2, 🚀 PVTv2
