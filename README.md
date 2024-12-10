# Salary Prediction Project  (Data Science - Python)

## 📄 Description

This project predicts the salary of an individual using **linear regression** and **multiple linear regression** models. The workflow includes importing a dataset, performing exploratory data analysis (EDA), training machine learning models, and evaluating their performance. The code is designed for ease of visualization, model interpretation, and deployment.

---

## 🔧 Technologies Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For basic data visualizations.
- **Seaborn**: For advanced statistical visualizations.
- **Scikit-learn**: For machine learning tasks like regression, model evaluation, and splitting data.
- **Joblib**: For saving and loading trained models.

---

## 🚀 Project Workflow

### Linear Regression

1. **Import the Dataset**:
   - Load the data from a CSV file.
   - Display basic information, summary statistics, and handle missing values.

2. **Exploratory Data Analysis (EDA)**:
   - **Distribution of Salary**: Visualize the distribution of the target variable using a histogram.
   - **Correlation Matrix**: Analyze variable correlations with a heatmap.

3. **Data Preprocessing**:
   - Select the independent (`YearsOfExperience`) and dependent (`Salary`) variables.
   - Split the dataset into training and testing sets (80/20 ratio).

4. **Train the Model**:
   - Use simple linear regression to model the relationship between years of experience and salary.

5. **Evaluate the Model**:
   - Calculate the **Mean Squared Error (MSE)** and **R² Score** to assess model performance.

6. **Make Predictions**:
   - Predict salaries using the trained model.
   - Provide a user input feature to predict salary based on a specific number of years of experience.

7. **Visualize Results**:
   - Plot actual vs. predicted salaries with a regression line.

8. **Save and Load the Model**:
   - Save the trained model using Joblib.
   - Load the saved model for future predictions.

---

### Multiple Linear Regression

1. **Feature Selection and Encoding**:
   - Features: `YearsOfExperience`, `Gender`, `EducationLevel`, `JobTitle`, and `Age`.
   - Encode categorical variables (e.g., Gender, EducationLevel) into numeric format using **Label Encoding**.

2. **EDA Enhancements**:
   - Pairplots with annotations for categorical variables.
   - Feature importance analysis using model coefficients.

3. **Model Training and Evaluation**:
   - Train a **multiple linear regression model** using the selected features.
   - Evaluate performance using **MSE** and **R² Score**.
   - Analyze feature importance and interpret the contribution of each factor to salary prediction.

4. **Advanced Visualizations**:
   - **Feature Importance Bar Plot**: Highlight key factors influencing salary.
   - **Error Analysis**: Visualize errors between actual and predicted salaries.

5. **Save and Load Advanced Model**:
   - Save the multiple linear regression model for deployment.
   - Load and test the saved model for predictions.

---

## 📊 Visualizations

1. **Distribution of Salary**:
   - Histogram with kernel density estimation for understanding salary distribution.

2. **Pairplot**:
   - Enhanced pairplot with hue for categorical differentiation and correlation annotations.

3. **Correlation Matrix**:
   - Heatmap with values, custom colormap, and masked upper triangle for clarity.

4. **Feature Importance Plot**:
   - Horizontal bar plot of regression coefficients with annotations directly on bars.

5. **Prediction Visualization**:
   - Scatter plot of actual vs. predicted salaries, including a regression line and error magnitudes.

---

## 📈 Model Performance

- **Linear Regression**:
  - **Mean Squared Error (MSE)**: Evaluates average prediction errors.
  - **R² Score**: Explains variance captured by the model.

- **Multiple Linear Regression**:
  - **Feature Importance**:
    - Analyzes the relative impact of factors such as `YearsOfExperience`, `Gender`, `EducationLevel`, etc., on salary.
    - Example:
      - `YearsOfExperience`: Most significant positive impact (+8985.79).
      - `Gender`: Notable positive impact (+6441.43).

---

## 📂 File Structure

- **`data/`**: Contains the dataset (`salary_dataset.csv`).
- **`plots/`**: Contains saved visualizations for linear regression.
- **`advanced_plots/`**: Contains saved visualizations for multiple linear regression.
- **`model/`**: Contains saved models:
  - **`salary_prediction_model.pkl`**: Linear regression model.
  - **`salary_prediction_ML_regression_model.pkl`**: Multiple linear regression model.
- **`main.py`**: Contains the project code for linear regression.
- **`ml_regression.py`**: Contains the code for multiple linear regression modeling.

---

## 📜 Further Reading

For an in-depth analysis of multiple linear regression, including feature importance, coefficients, and model insights, please refer to [README_ANALYSIS.md](README_ANALYSIS.md).
