# CNN Image Classification: From Scratch with PyTorch

This project demonstrates the full pipeline of designing, training, and evaluating a **Convolutional Neural Network (CNN)** for multi-class image classification from the ground up using PyTorch.  
The work involves custom data preprocessing, manual CNN design (without pre-trained models), performance optimization, and comprehensive evaluation using real-world datasets.

---

## 🔍 Project Overview

The objective of this project was to **classify images into multiple object categories** using a convolutional architecture trained entirely from scratch.  
The focus was on building a baseline CNN model, tuning its hyperparameters, and visualizing the model’s learning dynamics through accuracy and loss plots.

Unlike projects that use pre-trained architectures, this implementation emphasizes **deep understanding of CNN mechanics**, backpropagation behavior, and regularization techniques that help stabilize and improve performance.

---

## 🧠 Model Design

The CNN architecture was implemented manually in PyTorch and trained from scratch with no transfer learning or pre-trained weights.

**Architecture summary:**

| Layer | Type | Parameters / Details |
|-------|------|----------------------|
| 1 | `Conv2d` | 32 filters, 3×3 kernel, ReLU, padding = 1 |
| 2 | `MaxPool2d` | 2×2 stride |
| 3 | `Conv2d` | 64 filters, 3×3 kernel, ReLU |
| 4 | `Dropout` | p = 0.25 |
| 5 | `Flatten` | — |
| 6 | `Linear` | 128 → 64 units, ReLU |
| 7 | `Linear` | 64 → N_classes, Softmax |

**Loss & Optimization**
- **Loss function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Learning rate:** 0.001 (adaptive tuning)  
- **Regularization:** Dropout + Early Stopping  
- **Scheduler:** ReduceLROnPlateau for adaptive learning rate control  

---

## ⚙️ Data Pipeline

The dataset was preprocessed with the following transformations:

```python
transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```
Each subset (train/val/test) was loaded via torch.utils.data.DataLoader with mini-batches (batch = 32) for efficient GPU acceleration.
Dataset statistics, class distribution, and image samples were visualized to ensure balance and correctness.


## 📊 Training & Evaluation
The model was trained for 30 epochs with early stopping enabled after convergence.
| Metric   | Train               | Validation                 | Test       |
| -------- | ------------------- | -------------------------- | ---------- |
| Accuracy | 94.72 %             | 91.8 %                     | **87.6 %** |
| Loss     | steadily decreasing | plateaued after ~25 epochs | —          |

## 💡 Key Insights
- Regularization (dropout + scheduler) prevented overfitting in the deeper layers.
- Manual learning rate tuning was crucial in stabilizing training.
- The model captured fine texture and shape features efficiently despite limited epochs.
- Training from scratch provided interpretability over each layer’s contribution, a valuable learning experience beyond black-box fine-tuning.

## 🗝️ Results Summary
- Achieved 87.6 % test accuracy on a custom-curated dataset.
- Demonstrated smooth gradient propagation and stable learning.
- Achieved balanced predictions across classes with no major bias.
- Built a foundational CNN ready to be extended for VGG- and ResNet-style comparative studies (as done in Part 2).
