# Simple Linear Regression Project

This repository contains a simple Python project demonstrating the basics of Linear Regression using `pandas`, `numpy`, and `matplotlib` for data handling and visualization, and `scikit-learn` for model building and evaluation.

## Project Overview

This project aims to illustrate the fundamental steps involved in performing a simple linear regression:
1.  **Data Preparation**: Creating and inspecting a dataset.
2.  **Data Visualization**: Understanding the relationship between variables through plotting.
3.  **Model Building (Upcoming)**: Implementing a linear regression model.
4.  **Model Evaluation (Upcoming)**: Assessing the performance of the trained model.
5.  **Prediction (Upcoming)**: Using the trained model to make predictions.

## Files in This Repository

* `Untitled5.ipynb`: A Jupyter Notebook containing the Python code for data generation, initial data inspection, and visualization. This notebook serves as the core script for the project.

## Project Structure

The current notebook, `Untitled5.ipynb`, covers the following initial steps:

### 1. Importing Libraries
The project begins by importing necessary libraries for data manipulation, numerical operations, visualization, and machine learning:
* `pandas` as `pd`: For creating and managing DataFrames.
* `numpy` as `np`: For numerical operations and array handling.
* `matplotlib.pyplot` as `plt`: For creating static, interactive, and animated visualizations.
* `sklearn.model_selection.train_test_split`: To split data into training and testing sets (for future use).
* `sklearn.linear_model.LinearRegression`: To build the linear regression model (for future use).
* `sklearn.metrics.mean_squared_error, r2_score`: To evaluate model performance (for future use).

### 2. Data Creation
A fictitious dataset is created to simulate real-world data for house sizes and prices:
* The `data` dictionary holds `ukuran_rumah_m2` (house size in m²) and `harga_rumah_juta_rp` (house price in million IDR).
* This dictionary is then converted into a `pandas.DataFrame` named `df`.
* The first 5 rows of the DataFrame are printed using `df.head()`, and basic information (like data types and non-null counts) is displayed using `df.info()`.

### 3. Data Visualization
A scatter plot is generated to visually inspect the relationship between house size and house price:
* `df['ukuran_rumah_m2']` is plotted on the x-axis, and `df['harga_rumah_juta_rp']` on the y-axis.
* The plot includes a title, axis labels, and a grid for better readability.

## How to Run the Project

To run this project, you will need a Python environment with Jupyter Notebook installed.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Install the required libraries:**
    You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```
3.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook Untitled5.ipynb
    ```
4.  **Run the cells:**
    Once the notebook is open in your browser, you can run each cell sequentially to see the data creation and visualization.

## Future Enhancements (To Be Implemented)

* **Data Splitting**: Splitting the dataset into training and testing sets using `train_test_split`.
* **Model Training**: Training the `LinearRegression` model on the training data.
* **Prediction**: Making predictions on new (or test) data.
* **Model Evaluation**: Calculating `Mean Squared Error (MSE)` and `R-squared (R2)` to evaluate the model's performance.
* **Regression Line Visualization**: Adding the regression line to the scatter plot.

## Contribution

Feel free to fork this repository, make improvements, and submit pull requests.
