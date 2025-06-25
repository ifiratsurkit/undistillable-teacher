# undistillable-teacher
Project for BBL 514E: Implementing undistillable teacher models against knowledge distillation attacks 

# Undistillable Teacher: Resisting Knowledge Distillation Attacks

This project implements and evaluates a defensive strategy against knowledge distillation (KD)-based model stealing, inspired by "Undistillable Teachers" (Ma et al., ICLR 2021). It compares a standard teacher-student distillation pipeline with a modified "nasty" teacher model that resists being mimicked.

## 🔍 Project Structure
```
undistillable-teacher/
├── train_teacher.py              # Trains a standard ResNet-18 teacher on CIFAR-10
├── train_student.py              # Trains a student using soft-label distillation from standard teacher
├── train_nasty_teacher.py        # Trains a malicious (undistillable) teacher with misleading logits
├── train_student_from_nasty.py   # Trains a student using distillation from the undistillable teacher
├── checkpoints/                  # Stores .pth files for saved models
├── data/                         # Automatically downloaded CIFAR-10 dataset
└── README.md                     # Project overview and instructions
```

## ⚙️ Requirements
- Python 3.9+
- PyTorch
- torchvision
- numpy

Install dependencies:
```bash
pip install torch torchvision numpy
```

## 📦 Training and Evaluation

### 1. Train Standard Teacher
```bash
python train_teacher.py
```
Outputs: `checkpoints/teacher.pth`

### 2. Train Student from Teacher
```bash
python train_student.py
```
Outputs: `checkpoints/student.pth`

### 3. Train Nasty (Undistillable) Teacher
```bash
python train_nasty_teacher.py
```
Outputs: `checkpoints/nasty_teacher.pth`

### 4. Train Student from Nasty Teacher
```bash
python train_student_from_nasty.py
```
Outputs: `checkpoints/student_from_nasty.pth`

## 📊 Expected Results
- **Standard Distillation:** Student accuracy ~75–80%
- **From Nasty Teacher:** Accuracy drops significantly (e.g., ~60%)
- **Goal:** Demonstrate defense against model stealing via internal manipulation

## 📁 Dataset
- CIFAR-10 (60,000 images, 10 classes)
- Automatically downloaded with torchvision

## 🧪 Citation
> Ma, H., Chen, T., Hu, T. K., You, C., Xie, X., & Wang, Z. (2021). Undistillable: Making A Nasty Teacher That CANNOT Teach Students. ICLR.

---

Feel free to use this code as a baseline for experimenting with anti-distillation strategies.

© 2025 İsmail Fırat Sürkit

