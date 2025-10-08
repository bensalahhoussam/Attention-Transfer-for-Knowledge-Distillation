# Attention-Transfer-for-Knowledge-Distillation

PyTorch code for "Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer

<img width="320" height="382" alt="image" src="https://github.com/user-attachments/assets/94b7b3fc-f1ac-4e5e-9160-c2510ef6321b" />




## Experiments
code  includes:

* Activation-based spatial attention transfer implementation
* Knowledge distillation implementation
* Similarity-preserving knowledge distillation implementation

| Model | Resnet18 (F1) | Resnet101 (F1) |
|-----------|-----------|-----------|
| Focal | -   | 83.9   |
| Focal |  76.5  | -   |
| Focal + kd  |  85.3  | 83.9   |
| Focal + AT  |  88.8  | 83.9   |
| Focal + SP  |  84.7  | 83.9   |
