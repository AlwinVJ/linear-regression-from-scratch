# Linear Regression Evaluation Metrics

## Overview

After training the manually implemented `Linear_Regression` model, we
evaluate its predictions by comparing the actual test values (`Y_test`)
with the model predictions (`test_prediction`).

``` python
test_prediction = model.predict(X_test)

print("SSE :", sum_squared_error(Y_test, test_prediction))
print("MSE :", mean_squared_error(Y_test, test_prediction))
print("RMSE:", root_mean_squared_error(Y_test, test_prediction))
print("MAE :", mean_absolute_error(Y_test, test_prediction))
print("R²  :", r_squared(Y_test, test_prediction))
```

These metrics measure different aspects of regression error.

## 1. SSE --- Sum of Squared Errors

### Formula

$$
SSE = \sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

SSE adds all squared differences between actual and predicted values.

A smaller SSE means less total squared prediction error. However, SSE
increases with the number of observations, so it is mainly useful as a
total-error measure and in optimization.

``` python
def sum_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)
```

## 2. MSE --- Mean Squared Error

### Formula

$$
MSE = \frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

MSE is SSE divided by the number of observations:

$$
MSE = \frac{SSE}{m}
$$

It represents the average squared prediction error. Because errors are
squared, large errors have a greater influence.

For salary data, if salary is measured in rupees, MSE has units of
rupees squared, so it is less intuitive to interpret directly.

``` python
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

## 3. RMSE --- Root Mean Squared Error

### Formula

$$
RMSE = \sqrt{MSE}
$$

RMSE converts MSE back to the original target units.

If salary is measured in rupees:

``` text
Salary → ₹
RMSE   → ₹
```

For example, an RMSE of ₹30,000 means the prediction errors have a
typical magnitude around that scale, with larger errors receiving
greater influence.

``` python
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

## 4. MAE --- Mean Absolute Error

### Formula

$$
MAE = \frac{1}{m}\sum_{i=1}^{m}|y_i-\hat{y}_i|
$$

MAE is the average absolute difference between actual and predicted
values.

Unlike MSE and RMSE, it does not square the errors, so it is less
sensitive to large errors.

``` python
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

If:

``` text
MAE = ₹25,000
```

the average absolute difference between actual and predicted salary is
₹25,000 on the evaluated dataset.

## 5. R² --- R-Squared

### Formula

$$
R^2 = 1 - \frac{SSE}{SST}
$$

where:

$$
SST = \sum_{i=1}^{m}(y_i-\bar{y})^2
$$

R² measures how much of the variation in the target is accounted for by
the regression model relative to predicting the mean target value.

For example:

``` text
R² = 0.80
```

means the model accounts for 80% of the observed variation in the target
on that evaluation dataset.

R² is **not the same as prediction accuracy**.

``` python
def r_squared(y_true, y_pred):
    sse = sum_squared_error(y_true, y_pred)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (sse / sst)
```

In ordinary regression settings:

-   `R² = 1` → perfect predictions
-   `R² = 0` → equivalent to the mean-prediction baseline according to
    R²
-   `R² < 0` → worse than the mean-prediction baseline on the evaluated
    data

## Metric Comparison

  -----------------------------------------------------------------------
  Metric            Measures          Units             Large-error
                                                        sensitivity
  ----------------- ----------------- ----------------- -----------------
  **SSE**           Total squared     Target²           High
                    error                               

  **MSE**           Average squared   Target²           High
                    error                               

  **RMSE**          Root average      Target            High
                    squared error                       

  **MAE**           Average absolute  Target            Lower
                    error                               

  **R²**            Explained         Unitless          Indirect
                    variation                           
                    relative to mean                    
                    baseline                            
  -----------------------------------------------------------------------

## SSE → MSE → RMSE

These metrics are directly related:

``` text
Prediction errors
       ↓
Square errors
       ↓
      SSE
       ↓
Divide by number of observations
       ↓
      MSE
       ↓
Square root
       ↓
     RMSE
```

Therefore:

$$
MSE = \frac{SSE}{m}
$$

and:

$$
RMSE = \sqrt{MSE}
$$

## MAE vs RMSE

Both MAE and RMSE are expressed in the original target units, but they
respond differently to large errors.

**MAE:**

``` python
np.mean(np.abs(y_true - y_pred))
```

**RMSE:**

``` python
np.sqrt(np.mean((y_true - y_pred) ** 2))
```

RMSE gives greater influence to large errors because errors are squared
before averaging. MAE provides a more direct average absolute error.

## Applying the Metrics to the Salary Model

After training:

``` python
model.fit(X_train, Y_train)
```

Generate predictions on the held-out test data:

``` python
test_prediction = model.predict(X_test)
```

Then calculate the metrics:

``` python
print("SSE :", sum_squared_error(Y_test, test_prediction))
print("MSE :", mean_squared_error(Y_test, test_prediction))
print("RMSE:", root_mean_squared_error(Y_test, test_prediction))
print("MAE :", mean_absolute_error(Y_test, test_prediction))
print("R²  :", r_squared(Y_test, test_prediction))
```

Using `X_test` and `Y_test` is important because the test data was not
used to learn the model parameters.

## Key Takeaways

1.  **SSE** measures total squared prediction error.
2.  **MSE** measures average squared prediction error.
3.  **RMSE** expresses error in the original target units.
4.  **MAE** measures average absolute prediction error.
5.  **R²** measures explained variation relative to a mean-prediction
    baseline.
6.  MSE and RMSE are more sensitive to large errors than MAE.
7.  RMSE and MAE are particularly interpretable for salary prediction
    because they use salary units.
8.  R² should not be described as prediction accuracy.
9.  Multiple metrics provide a more complete view of regression
    performance.
10. Evaluation on held-out test data gives an indication of how the
    trained model performs on unseen observations.

## Evaluation Workflow

``` text
Training Data
     ↓
Train/Test Split
     ↓
Linear Regression
     ↓
Learn Weight + Bias
     ↓
Trained Model
     ↓
X_test
     ↓
Predictions
     ↓
Compare Y_test vs Y_pred
     ↓
SSE / MSE / RMSE / MAE / R²
```

This completes the basic evaluation stage of the from-scratch linear
regression project.
