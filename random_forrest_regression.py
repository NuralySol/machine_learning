
#! Random Forest Regression Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder 

# Load the data
data = pd.read_csv('./data/salary_dataset.csv')

# Check for missing values and drop them
print(f"Missing values: \n{data.isnull().sum()}")
data = data.dropna()

# Encode categorical variables
labelencoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
   data[column] = labelencoder.fit_transform(data[column])

# Split the data into independent ('X') and dependent ('y') variables
X = data.iloc[:, :-1].values  # Features (all columns except Salary)
y = data.iloc[:, -1].values   # Target variable (Salary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict the results
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Feature Names
feature_names = data.columns[:-1]  # All columns except Salary

# Create importance_df for Feature Importance
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
   'Feature': feature_names,
   'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Enhanced Visualization: Actual vs Predicted Values (Subset for Readability)
plt.figure(figsize=(12, 8))

# Use a subset of the test data for visualization
subset_size = 50  # Display only the first 50 samples
y_test_subset = y_test[:subset_size]
y_pred_subset = y_pred[:subset_size]

# Scatter plot for actual and predicted values
plt.scatter(range(len(y_test_subset)), y_test_subset, color='blue', alpha=0.7, s=60, label='Actual Values', edgecolor='black')
plt.scatter(range(len(y_pred_subset)), y_pred_subset, color='red', alpha=0.7, s=60, label='Predicted Values', edgecolor='black')

# Line plot for trends
plt.plot(range(len(y_test_subset)), y_test_subset, color='blue', linewidth=1.5, alpha=0.5, linestyle='--', label='Actual Trend')
plt.plot(range(len(y_pred_subset)), y_pred_subset, color='red', linewidth=1.5, alpha=0.5, linestyle='--', label='Predicted Trend')

# Title and axis labels
plt.title('Actual vs Predicted Values (Subset)', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Number of Test Samples (Subset)', fontsize=14)
plt.ylabel('Salary', fontsize=14)

# Customize grid and legend
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('./random_forrest_&_gradient_boosting/actual_vs_predicted_subset.png')
# Show the plot
plt.show()

# Enhanced Feature Importance Visualization
plt.figure(figsize=(12, 8))

# Bar plot for feature importance
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm', edgecolor='black')

# Add annotations on the bars
for bar in plt.gca().patches:
   bar_width = bar.get_width()
   plt.gca().text(
      bar_width + 0.01, bar.get_y() + bar.get_height() / 2, 
      f'{bar_width:.2f}', 
      ha='left', va='center', fontsize=12, color='black', weight='bold'  
   )

# Title and axis labels
plt.title('Feature Importance (Random Forest)', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)

# Customize grid and layout
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('./random_forrest_&_gradient_boosting/feature_importance.png')
# Show the plot
plt.show()

# Residuals Plot. The residuals plot is a scatter plot of the residuals (y_test - y_pred) against the predicted values (y_pred).
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residuals Plot', fontsize=16)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig('./random_forrest_&_gradient_boosting/residuals_plot.png')
# Show the plot
plt.show()

# Save the model to the project directory
joblib.dump(model, './model/random_forest_model.pkl')

#! Key Points in Random Forest Regression Model. 
"""
# Summary of Random Forest Regression Model

## Data Preprocessing:
1. **Missing Values**:
- The dataset was checked for missing values, and any rows containing null values were dropped.
- This ensures the model operates on clean data without introducing biases or errors.

2. **Categorical Encoding**:
- Non-numeric (categorical) columns were encoded using `LabelEncoder`.
- This converts features like `Gender`, `EducationLevel`, etc., into numerical format for compatibility with the regression model.

3. **Feature and Target Selection**:
- Independent features (`X`) include all columns except the target variable (`Salary`).
- The dependent target variable (`y`) is the `Salary`.

4. **Train-Test Split**:
- The dataset was split into training (80%) and testing (20%) subsets using `train_test_split` for model training and evaluation.

---

## Random Forest Regression:
1. **Model Creation**:
- A `RandomForestRegressor` with 100 estimators and a random seed (`random_state=0`) was trained on the training data.

2. **Model Evaluation**:
   - **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
     - **MSE Value**: Indicates the magnitude of prediction errors.
   - **R² Score**: Represents the proportion of variance in the target variable explained by the model.
     - **R² Value**: Higher values (close to 1) indicate better model performance.

   **Results**:
   - **MSE**: A low value implies good model accuracy.
   - **R² Score**: Indicates the model explained a significant portion of salary variance.

---

## Visualizations:
1. **Actual vs. Predicted Values**:
- A subset of test samples (e.g., 50) was used for a clear comparison of actual and predicted salaries.
   - **Scatter Plot**:
- Blue: Actual values.
- Red: Predicted values.
   - **Trend Lines**:
- Dashed lines depict trends for both actual and predicted values.
- **Purpose**: Provides insights into how well the model predicts salaries across test samples.

2. **Feature Importance**:
   - **Bar Plot**:
- Visualizes the relative importance of each feature in predicting salary.
- Features with higher importance have a stronger influence on the target variable.
   - **Annotations**:
- Display exact importance values directly on the bars for clarity.
   - **Purpose**: Helps identify which features contribute most to salary prediction.

---

## Insights:
1. **Key Features**:
- Features like `YearsOfExperience` likely dominate in importance, as seen from the feature importance plot.
- Categorical variables such as `Gender` or `EducationLevel` may also play significant roles.

2. **Model Interpretation**:
- The Random Forest model captures non-linear relationships between features and salary.
- The combination of multiple trees reduces overfitting, resulting in reliable predictions.

3. **Next Steps**:
- Fine-tune the model parameters (e.g., increase `n_estimators` or adjust `max_depth`) to further improve performance.
- Add more visualizations like residual plots to better understand the prediction errors.

---

## Conclusion:
This implementation of Random Forest Regression effectively models salary predictions using multiple features. It balances accuracy and interpretability, with clear visualizations to communicate results. The feature importance analysis provides actionable insights into which factors most influence salaries, making it suitable for real-world applications.
"""



#! MSE: 50493882.59 and R2: 0.98 Summary.
"""
# Model Performance Summary

## Evaluation Metrics:
1. **Mean Squared Error (MSE)**: 50,493,882.59
   - **Definition**: Measures the average squared difference between actual and predicted values.
   - **Interpretation**: 
   - A relatively low MSE indicates that the model's predictions are close to the actual salaries.
   - If salaries are in the range of tens or hundreds of thousands, this suggests good predictive performance.

2. **R² Score**: 0.98
   - **Definition**: Represents the proportion of variance in the target variable (Salary) explained by the model.
   - **Interpretation**:
     - A very high R² score means the model explains **98% of the variance** in salary data based on the features.
   - This indicates the model captures the relationships in the data exceptionally well.

## Key Takeaways:
- **Accuracy**: The low MSE and high R² score suggest that the model performs very well.
- **Variance Explanation**: The model explains almost all the variance in salaries, leaving very little unexplained.

## Recommendations:
- Use residual plots to analyze prediction errors and ensure there are no systematic patterns.
- Test the model on new or unseen data to confirm robustness.
"""