# Machine Learning Basics with PyTorch on CIFAR-10

Four classic classifiers implemented in **PyTorch** (GPU-accelerated) and validated on the CIFAR-10
image dataset with train/validation/test splits and hyperparameter tuning. The implementation is in
[`ml_basics_cifar10.ipynb`](ml_basics_cifar10.ipynb).

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
networks exploit.

## Running it

```bash
pip install torch torchvision numpy pandas matplotlib   # CUDA build of torch for GPU
jupyter notebook ml_basics_cifar10.ipynb
```

## Files

| File | Description |
|------|-------------|
| `ml_basics_cifar10.ipynb` | Full implementation and analysis (k-NN, SVM, linear and logistic regression) |
| `PROJECT_BRIEF.pdf` | Project brief (goals, objectives, outcomes) |
