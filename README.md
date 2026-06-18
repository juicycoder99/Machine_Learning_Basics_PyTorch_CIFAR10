# CS596 Research Topics — Assignment 1: Machine Learning Basics (CIFAR-10)

Coursework for **CS596 – Research Topics in Computer Science**, Bishop's University.

Four classic classifiers implemented in **PyTorch** (GPU-accelerated) and validated on the CIFAR-10
image dataset with train/validation/test splits and hyperparameter tuning. The solution is in
[`A1_ML_Basics_CIFAR10.ipynb`](A1_ML_Basics_CIFAR10.ipynb).

## Classifiers and results

Run on an NVIDIA RTX 3080. CIFAR-10 is downloaded automatically by `torchvision` on first run.

| Classifier | CIFAR-10 test accuracy |
|------------|-----------------------:|
| k-Nearest Neighbours (k tuned on validation, best k=1) | 0.354 |
| Linear SVM (multiclass hinge loss) | 0.276 |
| Logistic (softmax) regression | 0.276 |
| Linear regression (least-squares classifier) | 0.360 |

These are the expected baselines for classic classifiers on raw pixels: a single linear boundary or
a nearest-neighbour vote in pixel space cannot capture the visual structure that convolutional
networks (Assignment 2) exploit.

## Running it

```bash
pip install torch torchvision numpy pandas matplotlib   # CUDA build of torch for GPU
jupyter notebook A1_ML_Basics_CIFAR10.ipynb
```

## Files

| File | Description |
|------|-------------|
| `A1_ML_Basics_CIFAR10.ipynb` | Full solution (k-NN, SVM, linear & logistic regression) |
| `Assignment 1 Machine Learning Basics-converti.pdf` | Assignment description |
