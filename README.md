# Neural Networks, Regression Analysis, and Log Likelihood Calculation

## Project Overview
Here, we developed an artificial neural network for classifying data from the Dry Beans Dataset and perform various regression analyses. The project includes exercises on polynomial regression, log-likelihood calculation, and the application of gradient descent for model fitting.

## Files in the Repository
- **Neural_Networks_and_Regression_Analysis.ipynb**: This Jupyter notebook contains all the exercises.
- **auto-mpg.csv**: Dataset used for initial data exploration and analysis.
- **cost.csv**: Dataset used for polynomial regression analysis.
- **Dry_Beans_Dataset.csv**: Dataset used for neural network classification.

## How to Use
1. **Prerequisites**:
   - Python 3.x
   - Jupyter Notebook or JupyterLab
   - Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` or `keras`

2. **Installation**:
   Ensure you have the required packages installed. You can install them using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

3. **Running the Notebook**:
   - Open the Jupyter Notebook:
     ```bash
     jupyter notebook Homework_2_Neural_Networks_and_Regression_Analysis.ipynb
     ```
   - Execute the cells in the notebook sequentially to perform the various tasks and analyses.

## Sections in the Notebook

### 1. Homework 2 Introduction
This section introduces the assignment, outlining the datasets used (Dry Beans, auto-mpg, and cost datasets) and the key tasks to be performed, including neural network classification, polynomial regression, and log-likelihood calculation.

### 2. Exercise 1 - Neural Network Classification
#### Description:
Develop an artificial neural network to classify the Dry Beans Dataset.
#### Key Steps:
   - Load and preprocess the Dry Beans Dataset.
   - Build and compile the neural network model using TensorFlow/Keras.
   - Train the model and evaluate its performance.
   - Visualize the training history and classification results.

### 3. Exercise 2 - Exploratory Data Analysis
#### Description:
Perform exploratory data analysis on the auto-mpg dataset.
#### Key Steps:
   - Load and preprocess the auto-mpg dataset.
   - Generate descriptive statistics and visualizations.
   - Analyze correlations and distributions of the variables.

### 4. Exercise 3 - Polynomial Regressor using Gradient Descent
#### Description:
Perform polynomial regression on the cost dataset and analyze the results.
#### Key Steps:
   - Load and split the cost dataset into training and testing sets.
   - Implement polynomial regression with degrees 1, 2, 3, and 4.
   - Compute RMSE and R² for training and testing sets.
   - Identify the best polynomial degree based on performance metrics.
   - Visualize the fitted models.

### 5. Exercise 4 - Log Likelihood Calculation
#### Description:
Calculate the log-likelihood for a model with two independent variables using a given dataset.
#### Key Steps:
   - Load the dataset with weight and length as independent variables, and height as the dependent variable.
   - Compute the predicted height using the given model.
   - Calculate squared residuals and log-likelihood for each data point.
   - Fill in the provided table with the computed values.

## Visualization
The notebook includes various visualizations to support the analysis, such as correlation heatmaps, scatter plots, polynomial regression curves, and training history plots. Each section's visualizations help in understanding the data and the results of the applied techniques.

## Conclusion
This notebook provides a comprehensive approach to data analysis using neural networks, polynomial regression, and log-likelihood calculations. By following the steps in the notebook, users can replicate the analyses on similar datasets or extend them to other data.

If you have any questions or encounter any issues, please feel free to reach out for further assistance.