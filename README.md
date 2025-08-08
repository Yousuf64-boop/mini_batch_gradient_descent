# Mini-Batch Gradient Descent for Linear Regression

This project implements **Mini-Batch Gradient Descent (MBGD)** from scratch in Python using NumPy, applied to Linear Regression. It demonstrates how model parameters (weights and bias) are updated using randomly selected subsets (mini-batches) of the training data.

## 📄 Notebook

- `mini_batch_gradient_descent.ipynb`: Contains the full implementation of the `MBGDRegressor` class and example usage with synthetic or real data.

## 🚀 Highlights

- Custom linear regression class `MBGDRegressor`
- Mini-batch based parameter updates
- Gradient computation and loss minimization using Mean Squared Error (MSE)
- Easily configurable:
  - Batch size
  - Learning rate
  - Number of epochs

## 🧠 How It Works

1. **Initialization**:
   - Weights (`coef_`) are initialized to ones.
   - Bias (`intercept_`) is initialized to zero.

2. **Training (fit)**:
   - For each epoch:
     - Random mini-batches of data are selected using `random.sample`.
     - Predictions (`y_hat`) are calculated using the linear model.
     - Gradients are computed with respect to weights and bias.
     - Parameters are updated using the learning rate.

3. **Prediction**:
   - Predicts values using the learned linear model:  
     \[
     \hat{y} = X \cdot \text{coef\_} + \text{intercept\_}
     \]

## 🛠️ Requirements

- Python 3.x
- NumPy

To install NumPy:

```bash
pip install numpy
