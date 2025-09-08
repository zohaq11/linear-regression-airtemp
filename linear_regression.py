# CSC311 Lab 3: Linear Regression

"""
In this lab, we will build a linear regression model to predict the air temperature at the UTM pond
given information collected at the UTM forest.

Acknowledgements
    Data is from https://www.utm.utoronto.ca/geography/resources/meteorological-station/environmental-datasets
"""

import matplotlib.pyplot as plt  # For plotting
import numpy as np               # Linear algebra library
import pandas as pd              # For data manipulation
from sklearn.linear_model import LinearRegression
import os
import urllib.request
import zipfile

def download_data():
    """Download and extract data files if they don't already exist."""
    data_dir = "data"
    data_url = "https://www.cs.toronto.edu/~lczhang/311/lab03/data.zip"
    zip_file = "data.zip"
    
    # Check if data directory exists and has the required files
    required_files = [
        "data2016sept.csv",
        "data2016oct.csv", 
        "data2017sept.csv",
        "data2017oct.csv"
    ]
    
    data_files_exist = all(os.path.exists(os.path.join(data_dir, file)) for file in required_files)
    
    if not data_files_exist:
        print("Data files not found. Downloading and extracting data...")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Download the zip file
        print(f"Downloading {data_url}...")
        urllib.request.urlretrieve(data_url, zip_file)
        
        # Extract the zip file
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall()
        
        # Clean up the zip file
        os.remove(zip_file)
        print("Data download and extraction completed!")
    else:
        print("Data files already exist. Skipping download.")

def main():
    """Main function to run the linear regression analysis."""
    
    # Part 1. Data
    print("DATA")
    
    # Download data files if they don't exist
    download_data()
    
    # Read each of the csv files as a pandas data frame
    try:
        data201610 = pd.read_csv('data/data2016oct.csv')
        data201609 = pd.read_csv('data/data2016sept.csv')
        data201710 = pd.read_csv('data/data2017oct.csv')
        data201709 = pd.read_csv('data/data2017sept.csv')
        
        print("Data loaded successfully!")
        print("Sample data from 2016 September:")
        print(data201609.head())
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please check your internet connection and try again")
        return
    
    # Display scatter plots for forest features
    print("\nGenerating scatter plots for forest features...")
    data_set = data201609  # change me
    for fet in ["forest_air_temp_c", "forest_soil_temp_c", "forest_rh", "forest_soil_wc"]:
        plt.figure()
        X = data_set[fet]
        t = data_set["pond_air_temp_c"]
        plt.title("UTM Environmental Data in 2016 Sept")
        plt.scatter(X, t)
        plt.xlabel(fet)
        plt.ylabel("pond_air_temp_c")
        plt.show()
    
    # Data Splitting
    print("\nData Splitting")
    """
    NOTE: Rather than choosing a random percentage of data points to leave out in our test set, 
    we will instead place the most recent data points in our test set.

    This is because the data is based on/affected by the time of the year and so separating data 
    points of a consecutive time period makes more sense than at random times.
    """
    
    train_data = pd.concat([data201609, data201610])
    valid_data = data201709 # second most recent data points
    test_data = data201710 # most recent data points

    print("Number of training examples:", len(train_data)) 
    print("Number of validation examples:", len(valid_data))
    print("Number of test examples:", len(test_data))

    # Generate data matrices
    X_train, t_train = get_input_targets(train_data)
    X_valid, t_valid = get_input_targets(valid_data)
    X_test, t_test = get_input_targets(test_data)
    
    print(f"X_train shape: {X_train.shape}") # should be (N, 5)
    print(f"X_valid shape: {X_valid.shape}") 
    print(f"X_test shape: {X_test.shape}")
    
    # Part 2. Linear Regression Model
    print("\nLINEAR REGRESSION MODEL")

    # NOTE: Linear Regression Model is of the form:
    # y = f(x) = w^T x
    # where y is the prediction, x is a vector consisting of our features (input data), and w is a vector of the trainable weights (bias term included).

    # Test the gradient function with finite differences
    print("Testing gradient function with finite differences...")
    h = 0.0001  # a reasonably small value of "h"
    w = np.ones(5)   # a vector of weights
    for j in [0, 1, 2, 3, 4]:
        perturbed_w = np.copy(w)  # start by making a copy of w
        perturbed_w[j] += h  # perturb the jth element of perturbed_w by h
        
        estimate = (mse(perturbed_w, X_train, t_train) - mse(w, X_train, t_train)) / h
        
        print("Gradient Checking for weight j=", j)
        print("grad(w)[j]", grad(w, X_train, t_train)[j])
        print("(mse(perturbed_w) - mse(w)) / h", estimate)
    
    # Test different learning rates
    print("\nTesting learning rate that is too low...") # low alpha
    solve_via_gradient_descent(alpha=0.0000001, niter=500, X_train=X_train, t_train=t_train, X_valid=X_valid, t_valid=t_valid)
    
    print("\nTesting learning rate that is too high...") # high alpha
    solve_via_gradient_descent(alpha=0.01, niter=500, X_train=X_train, t_train=t_train, X_valid=X_valid, t_valid=t_valid)
    
    # Best hyperparameters
    alpha = 0.0001  # chosen value
    niter = 500     # chosen value
    print("\nUsing best hyperparameters: alpha =", alpha, "and niter =", niter)
    best_w = solve_via_gradient_descent(alpha=alpha, niter=niter, X_train=X_train, t_train=t_train, X_valid=X_valid, t_valid=t_valid)
    
    # Part 3. Linear Regression via sklearn
    print("\nLINEAR REGRESSION VIA SKLEARN")
    
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train, t_train)
    
    print("sklearn weights:", lr.coef_)
    print("Training MSE:", mse(lr.coef_, X_train, t_train))
    print("Validation MSE:", mse(lr.coef_, X_valid, t_valid))
    # lower training MSE and lower validation MSE compared to custom model
    
    # Part 4. Reporting Test Accuracy
    print("\nREPORTING TEST ACCURACY")
    """
    We should choose the model that gives the lowest validation MSE
    by using the validation set for model selection and the test set only for
    final evaluation, we ensure that our performance metrics are unbiased and
    our model generalizes well to truly unseen data
    """
    
    # Compare models and report test accuracy
    custom_train_mse = mse(best_w, X_train, t_train)
    custom_valid_mse = mse(best_w, X_valid, t_valid)
    sklearn_train_mse = mse(lr.coef_, X_train, t_train)
    sklearn_valid_mse = mse(lr.coef_, X_valid, t_valid)
    # NOTE: overfitting when training MSE is low and validation MSE is high
    
    print(f"\nCustom model - Training MSE: {custom_train_mse:.6f}, Validation MSE: {custom_valid_mse:.6f}")
    print(f"sklearn model - Training MSE: {sklearn_train_mse:.6f}, Validation MSE: {sklearn_valid_mse:.6f}")
    
    # Choose the better model based on validation MSE
    if sklearn_valid_mse < custom_valid_mse:
        best_model_weights = lr.coef_
        best_model_name = "sklearn"
    else:
        best_model_weights = best_w
        best_model_name = "custom"
    
    print(f"\nBest model: {best_model_name}")
    test_mse = mse(best_model_weights, X_test, t_test)
    print(f"Test MSE: {test_mse:.6f}")

def get_input_targets(data):
        """
        Produces the data matrix and target vector given a dataframe `data` read
        from one of the csv files containing the UTM weather data.

        The returned data matrix should have a column of 1's, so that the bias
        parameter will be folded into the weight vector.
        """
        # extract the target vector
        t = np.array(data["pond_air_temp_c"])

        # extract the data matrix:
        X_fets = np.array(data[["forest_air_temp_c", "forest_soil_temp_c", "forest_rh", "forest_soil_wc"]])

        n = len(data) # number of data points

        X = np.concatenate([np.ones((n, 1)), X_fets], axis=1) # add a column of 1's for the bias term

        return (X, t)

def pred(w, X):
    """
    Compute the prediction made by a linear hypothesis with weights `w`
    on the data set with input data matrix `X`. Recall that N is the number of
    samples and D is the number of features. The +1 accounts for the bias term.

    Parameters:
        `w` - a numpy array of shape (D+1)
        `X` - data matrix of shape (N, D+1)

    Returns: prediction vector of shape (N)
    """
    # NOTE: The prediction is given by:
    # y = Xw
    # where X is the data matrix, w is the weight vector, and y is the prediction vector.
    # The bias term is included in the weight vector.

    return np.dot(X, w) # (N, D+1) * (D+1, 1) = (N, 1) = (N,) = prediction vector, bias term included

def mse(w, X, t):
    """
    Compute the mean squared error of a linear hypothesis with weights `w`
    on the data set with input data matrix `X` and targets `t`

    Parameters:
        `weight` - a numpy array of shape (D+1)
        `X` - data matrix of shape (N, D+1)
        `t` - target vector of shape (N)

    Returns: a scalar MSE value
    """
    # NOTE: The MSE is given by:
    # MSE = 1/2N * Σ(y - t)^2
    # where y is the prediction, t is the target, and N is the number of data points.

    n =  X.shape[0] # the number of data points (N,)
    y = pred(w, X)  # the vector of predictions (N,)
    return np.sum((y-t) ** 2) / (2 * n) # compute the MSE in a vectorized way (N,)

def grad(w, X, t):
    '''
    Return gradient of the cost function at `w`. The cost function
    is the average square loss (MSE) across the data set X and the
    target vector t.

    Parameters:
        `weight` - a current "guess" of what our weights should be,
                   a numpy array of shape (D+1)
        `X` - matrix of shape (N,D+1) of input features
        `t` - target y values of shape (N)

    Returns: gradient vector of shape (D+1)
    '''
    # NOTE: The gradient of the cost function is given by:
    # ∇J(w) = 1/N * X^T * (y - t)
    # where X is the data matrix, y is the predictions, and t is the targets.

    n = X.shape[0] # the number of data points (N,)
    y = pred(w, X) # the vector of predictions (N,)
    return np.dot(X.T, (y - t)) / n # (D+1, N) * (N,) = (D+1,)

def solve_via_gradient_descent(X_train, t_train, X_valid, t_valid, 
                               alpha=0.0025, niter=1000, w_init=None):
    '''
    Given `alpha` - the learning rate
          `niter` - the number of iterations of gradient descent to run
          `X_train` - the data matrix to use for training
          `t_train` - the target vector to use for training
          `X_valid` - the data matrix to use for validation
          `t_valid` - the target vector to use for validation
          `w_init` - the initial `w` vector (if `None`, use a vector of all zeros)
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    '''
    # initialize all the weights to zeros
    if w_init is None:
        w = np.zeros(X_train.shape[1])
    else:
        w = w_init

    # we will track the MSE value at each iteration to record progress
    train_mses = []
    valid_mses = []

    for it in range(niter):
        # update `w` using the gradient descent update rule
        # w = w - alpha * ∇J(w)
        w = w - alpha * grad(w, X_train, t_train)
        # Record the current training and validation MSE values
        # Note that in practice, it is expensive to compute MSE at
        # every iteration, and practitioners will typically compute cost
        # every few iterations instead (e.g. every ~10, 100 or 1000 iterations,
        # depending on the learning task)
        train_mse = mse(w, X_train, t_train)
        valid_mse = mse(w, X_valid, t_valid)
        train_mses.append(train_mse)
        valid_mses.append(valid_mse)

    plt.title("Training Curve Showing Training and Validation MSE at each Iteration")
    plt.plot(train_mses, label="Training MSE")
    plt.plot(valid_mses, label="Validation MSE")
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    print("Final Training MSE:", train_mses[-1])
    print("Final Validation MSE:", valid_mses[-1])

    return w

if __name__ == "__main__":
    main()
