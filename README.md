# Linear Regression From Scratch in Python

A practical implementation of **Linear Regression from scratch using Python and NumPy**.

The goal of this project is not to use a pre-built machine learning implementation, but to understand what happens internally when a linear regression model is trained.

The implementation will manually handle:

* Model parameters (`weights` and `bias`)
* Predictions
* Loss calculation
* Gradients
* Gradient Descent
* Parameter updates
* Model training
* Predictions on new data

The project is based on the concepts demonstrated in a machine learning course video where a linear regression model is implemented from scratch using Python and NumPy.

## Objective

Build a working linear regression model without using a pre-built implementation such as:

```python
from sklearn.linear_model import LinearRegression
```

Instead, the core learning algorithm will be implemented manually.

The final model should be capable of learning a relationship such as:

```text
Salary = weight × YearsOfExperience + bias
```

and using the learned parameters to make predictions.

---

## Mathematical Foundation

The model is represented as:

$$
\hat{y} = wx + b
$$

where:

* `x` = input feature
* `w` = weight
* `b` = bias
* `ŷ` = predicted value

For multiple features, the model becomes:

$$
\hat{y} = Xw + b
$$

The model parameters are learned using **Gradient Descent**.

The parameters are updated using:

$$
w := w - \alpha \frac{\partial J}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

where:

* `α` = learning rate
* `J` = cost function
* `∂J/∂w` = gradient with respect to the weight
* `∂J/∂b` = gradient with respect to the bias

---

## Project Structure

```text
linear-regression-from-scratch/
│
├── README.md
├── environment.yml
├── .gitignore
│
├── data/
│   └── salary_data.csv
│
├── src/
│   ├── __init__.py
│   └── linear_regression.py
│
├── notebooks/
│   └── experiments.ipynb
│
└── tests/
    └── test_linear_regression.py
```

### `src/`

Contains the actual implementation of the linear regression algorithm.

### `data/`

Contains the dataset used for training and evaluating the model.

### `notebooks/`

Used for experimentation, visualization, and understanding how the algorithm behaves.

### `tests/`

Contains tests to verify that our implementation works correctly.

---

## Environment Setup

This project uses **Conda**.

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate lr-from-scratch
```

Verify Python:

```bash
python --version
```

Verify NumPy:

```bash
python -c "import numpy; print(numpy.__version__)"
```

---

## Planned Implementation

The linear regression model will be implemented as a Python class:

```python
class LinearRegression:
    ...
```

The class will contain methods for:

```text
__init__()
    ↓
fit()
    ↓
gradient descent
    ↓
update parameters
    ↓
predict()
```

The initial implementation will contain:

### 1. Initialization

Set the model's hyperparameters:

```text
learning_rate
iterations
```

### 2. Parameter Initialization

Initialize:

```text
weights
bias
```

### 3. Prediction

Calculate:

```text
ŷ = Xw + b
```

### 4. Gradient Calculation

Calculate the gradients:

```text
dw
db
```

based on the difference between predicted and actual values.

### 5. Gradient Descent

Update the parameters repeatedly:

```text
w = w - learning_rate × dw
b = b - learning_rate × db
```

### 6. Training

Repeat the gradient-descent process for the specified number of iterations.

### 7. Prediction

After training, use the learned parameters to predict values for new inputs.

---

## Dataset

The initial experiment will use a simple dataset containing:

```text
Years of Experience → Salary
```

The objective is to learn the relationship between years of experience and salary.

Example:

```text
Years of Experience
        ↓
Linear Regression Model
        ↓
Predicted Salary
```

---

## Learning Approach

This project will be developed incrementally.

### Phase 1 — Mathematical Foundation

Understand:

* Linear equation
* Weight
* Bias
* Prediction
* Loss / cost
* Derivatives
* Partial derivatives
* Gradients
* Gradient descent
* Learning rate

### Phase 2 — NumPy Implementation

Implement the mathematical operations using NumPy.

### Phase 3 — Linear Regression Class

Build:

```python
LinearRegression
```

with:

```python
fit()
predict()
```

and the internal gradient-descent mechanism.

### Phase 4 — Training

Train the model using the salary dataset.

### Phase 5 — Visualization

Visualize:

* Training data
* Regression line
* Predictions
* Residuals
* Cost over iterations

### Phase 6 — Validation

Compare our implementation with a trusted implementation such as scikit-learn **only after the from-scratch implementation is complete**.

The comparison will be used for verification rather than for implementing the model.

---

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter
* pytest
* Conda

Scikit-learn will not be used to implement the model itself.

---

## Learning Goal

The main goal of this project is to understand what happens inside a linear regression algorithm rather than treating machine learning models as black boxes.

By the end of the project, the complete training process should be understandable as:

```text
Training Data
     ↓
Initialize w and b
     ↓
Make Predictions
     ↓
Calculate Error
     ↓
Calculate Gradients
     ↓
Update w and b
     ↓
Repeat
     ↓
Convergence
     ↓
Trained Linear Regression Model
     ↓
Make Predictions
```

---

## Status

**Project initialized**

Current stage:

```text
[✓] Project structure
[✓] Conda environment configuration
[ ] Dataset
[ ] Linear Regression implementation
[ ] Gradient Descent
[ ] Training
[ ] Visualization
[ ] Testing
[ ] Comparison with scikit-learn
```

---

## Important Note

This project intentionally focuses on understanding the algorithm internally.

The implementation will therefore avoid using:

```python
sklearn.linear_model.LinearRegression
```

for the actual model training.

NumPy is used for numerical and matrix operations.