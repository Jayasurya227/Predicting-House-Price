# Predicting Housing Market Trends with AI 🏠

This project focuses on building a regression model to accurately predict the sale price of houses using the popular Kaggle "House Prices: Advanced Regression Techniques" dataset. It serves as a comprehensive demonstration of a complete machine learning workflow, suitable for showcasing data science skills in a professional portfolio.

The analysis covers essential stages from exploratory data analysis (EDA) and target variable transformation to advanced preprocessing, feature engineering, model training (comparing baseline and advanced models), and evaluation.

**Dataset:** `train.csv`, `test.csv` (from Kaggle's House Prices competition)
**Focus:** Demonstrating robust data processing, feature engineering, regression modeling, and evaluation techniques for a predictive task.
**Repository:** [https://github.com/Jayasurya227/Predicting-House-Price](https://github.com/Jayasurya227/Predicting-House-Price)

***

## Key Techniques & Concepts Demonstrated

Based on the analysis within the notebook (`3. Predicting Housing Market Trends with AI.ipynb`), the following key machine learning concepts and techniques are applied:

* **Target Variable Analysis:** Identified and addressed positive skewness in `SalePrice` using a **log transformation** (`np.log1p`) to normalize the distribution, benefiting model performance.
* **Correlation Analysis:** Explored relationships between numerical features and the target variable, identifying top predictors like `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, etc., using a correlation heatmap.
* **Advanced Data Preprocessing:** Implemented robust strategies for handling missing values across numerous features, using context-appropriate imputation methods:
    * Filling with **0** or **'None'** where NA implies absence (e.g., `MasVnrArea`, `PoolQC`).
    * Imputing `LotFrontage` with the **median value grouped by neighborhood**.
    * Imputing other categorical/numerical missing values with the **mode** or **0**.
* **Feature Engineering:** Created new, potentially more predictive features from existing ones:
    * `TotalSF`: Combined basement, 1st floor, and 2nd floor square footage.
    * `TotalBath`: Consolidated full and half bathrooms above and below grade.
    * `Age`: Calculated the age of the house at the time of sale (`YrSold` - `YearBuilt`).
* **Categorical Encoding:** Applied **One-Hot Encoding** (`pd.get_dummies`) to convert nominal categorical features into a numerical format suitable for modeling.
* **Feature Scaling:** Utilized **StandardScaler** from Scikit-learn to standardize numerical features, essential for models like Linear Regression.
* **Model Building & Comparison:** Trained and evaluated two distinct regression models:
    * A baseline **Linear Regression** model (using scaled data).
    * An advanced **XGBoost Regressor** model (using unscaled data, as tree-based models are less sensitive to feature scale).
* **Model Evaluation:** Compared model performance using standard regression metrics: **RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error), and **R-squared** ($R^2$), demonstrating the superior performance of XGBoost.
* **Prediction & Submission:** Generated predictions on the test set using the best model (XGBoost), reversed the initial log transformation (`np.expm1`) to get predictions on the original price scale, and formatted the output into a Kaggle submission file (`submission.csv`).

***

## Analysis Workflow

The notebook (`3. Predicting Housing Market Trends with AI.ipynb`) follows a structured machine learning process:

1.  **Setup & Data Loading:** Importing necessary libraries (Pandas, NumPy, Scikit-learn, XGBoost, etc.) and loading the train/test datasets directly via the Kaggle API.
2.  **Target Variable EDA & Transformation:** Analyzing the `SalePrice` distribution, identifying skewness, and applying a log transformation (`np.log1p`).
3.  **Feature EDA:** Calculating and visualizing the correlation matrix for the top 10 features most correlated with `SalePrice`.
4.  **Data Preprocessing:**
    * Combining train and test sets for consistent processing.
    * Implementing comprehensive **missing value imputation** strategies for both numerical and categorical features based on their meaning.
5.  **Feature Engineering:** Creating `TotalSF`, `TotalBath`, and `Age` features to capture combined or derived information.
6.  **Categorical Encoding:** Applying one-hot encoding to all object-type columns.
7.  **Model Building:**
    * Splitting the enhanced data back into training (X, y) and final test sets (X_test_final).
    * Performing a train-validation split on the training data.
    * Applying feature scaling (`StandardScaler`) to the numerical features for the Linear Regression model.
    * Training both Linear Regression (baseline) and XGBoost Regressor models.
8.  **Model Evaluation:** Calculating RMSE, MAE, and R-squared on the validation set predictions for both models to compare performance.
9.  **Prediction & Submission File Generation:** Using the trained XGBoost model to predict on the (unscaled) final test set, reversing the log transformation on predictions, and creating the `submission.csv` file in the required format.

***

## Technologies Used

* **Python**
* **Pandas & NumPy:** For data loading, manipulation, cleaning, and feature engineering.
* **Matplotlib & Seaborn:** For data visualization (histograms, heatmaps).
* **Scikit-learn:** For data splitting (`train_test_split`), scaling (`StandardScaler`), Linear Regression modeling (`LinearRegression`), and evaluation metrics (`mean_squared_error`, `r2_score`, `mean_absolute_error`).
* **XGBoost:** For advanced gradient boosting regression modeling (`XGBRegressor`).
* **SciPy:** For statistical functions (like `skew`).
* **Kaggle API:** For downloading the dataset directly.
* **Jupyter Notebook / Google Colab:** For the interactive analysis environment.

***

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jayasurya227/Predicting-House-Price.git](https://github.com/Jayasurya227/Predicting-House-Price.git)
    cd Predicting-House-Price
    ```
2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy kaggle jupyter
    ```
3.  **Set up Kaggle API:**
    * Go to your Kaggle account page (`https://www.kaggle.com/account`) and click 'Create New Token' under the API section to download `kaggle.json`.
    * Place the `kaggle.json` file in the expected location (e.g., `~/.kaggle/kaggle.json` on Linux/Mac). If using Google Colab, the notebook prompts for an upload.
    * Ensure the file has the correct permissions (`chmod 600 ~/.kaggle/kaggle.json`).
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "3. Predicting Housing Market Trends with AI.ipynb"
    ```
    *(Run the cells sequentially. The notebook handles dataset download via the Kaggle API.)*

***

## Author & Portfolio Use

* **Author:** Jayasurya227
* **Portfolio:** This project ([https://github.com/Jayasurya227/Predicting-House-Price](https://github.com/Jayasurya227/Predicting-House-Price)) demonstrates a comprehensive regression analysis and machine learning workflow, making it suitable for showcasing skills on GitHub, resumes/CVs, LinkedIn, and during data science interviews.
* **Notes:** Recruiters can review the detailed methodology for data cleaning, feature engineering, handling transformations, model selection (baseline vs. advanced), and evaluation implemented in this notebook. The process highlights practical considerations in building a predictive model.
