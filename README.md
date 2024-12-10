# Salary Prediction Project

## 📄 Description

This project predicts the salary of an individual based on their years of experience using a linear regression model. The workflow includes importing a dataset, performing exploratory data analysis (EDA), training a machine learning model, and evaluating its performance. The code is designed for ease of visualization and model deployment.

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

1. **Import the Dataset**:
   - Load the data from a CSV file.
   - Display basic information, summary statistics, and handle missing values.

2. **Exploratory Data Analysis (EDA)**:
   - **Distribution of Salary**: Visualize the distribution of the target variable using a histogram.
   - **Pairplot with Annotations**: Explore relationships between variables with pairwise plots, including correlation coefficients.
   - **Correlation Matrix**: Analyze variable correlations with a heatmap.

3. **Data Preprocessing**:
   - Select the independent (`YearsOfExperience`) and dependent (`Salary`) variables.
   - Split the dataset into training and testing sets (80/20 ratio).

4. **Train the Model**:
   - Use linear regression to model the relationship between years of experience and salary.

5. **Evaluate the Model**:
   - Calculate the **Mean Squared Error (MSE)** and **R² Score** to assess model performance.

6. **Make Predictions**:
   - Predict salaries using the trained model.
   - Provide a user input feature to predict salary based on a specific number of years of experience.

7. **Visualize Results**:
   - Plot actual vs. predicted salaries with a regression line.
   - Annotate and enhance all visualizations for professional presentation.

8. **Save and Load the Model**:
   - Save the trained model using Joblib.
   - Load the saved model for future predictions.

---

## 📊 Visualizations

1. **Distribution of Salary**:
   - Histogram with kernel density estimation for understanding salary distribution.

2. **Pairplot**:
   - Enhanced pairplot with hue for categorical differentiation and correlation annotations.

3. **Correlation Matrix**:
   - Heatmap with values, custom colormap, and masked upper triangle for clarity.

4. **Prediction Visualization**:
   - Scatter plot of actual vs. predicted salaries, including a regression line.

---

## 📈 Model Performance

- **Mean Squared Error (MSE)**: Evaluates average prediction errors.
- **R² Score**: Explains variance captured by the model.

---

## 📂 File Structure

- **`data/`**: Contains the dataset (`salary_dataset.csv`).
- **`plots/`**: Contains saved visualizations.
- **`model/`**: Contains the saved regression model (`salary_prediction_model.pkl`).
- **`main.py`**: Contains the project code.

---
