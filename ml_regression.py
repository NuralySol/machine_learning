import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder 

# Load the data and display the first 5 rows of the data set which is a 2D array.
data = pd.read_csv('./data/salary_dataset.csv')
print(f"Head of the data: \n{data.head()}")
print("-" * 100)

# Check for missing values in the data set
print(f"Missing values: \n{data.isnull().sum()}")
data = data.dropna()  # Drop missing values
print("-" * 100)

# Encode categorical variables using LabelEncoder from sklearn.
labelencoder = LabelEncoder()

# Automatically encode all non-numeric columns from the data set. 0, 1, 2, ... are assigned to each unique value.
for column in data.select_dtypes(include=['object']).columns:
    data[column] = labelencoder.fit_transform(data[column])
    print(f"Encoded column: {column}")
print("-" * 100)

# Display the first few rows to confirm encoding
print(data.head())
print("-" * 100)

# Split the data into independent ('X') and dependent ('y') variables
X = data.iloc[:, :-1].values  # All columns except the last as features
y = data.iloc[:, -1].values   # Last column as the target variable (Salary in this case)

# Split the data into training and testing sets. Random state is set to 0 to ensure reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the model using the training data. 
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the results using the testing data. 
y_pred = model.predict(X_test)

# Get the model's coefficients and intercept (bias), and display them. 
print(f"Slope (coefficients): {model.coef_}")
print(f"Intercept: {model.intercept_}")
print("-" * 100)

# Evaluate the model using Mean Squared Error and R2 Score and display the results.
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
print("-" * 100)

# Enhanced Visualization for Predictions vs. Actual Values
plt.figure(figsize=(12, 6))

# Add a diagonal line representing perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')

# Scatter plot for actual vs predicted values with color gradient for error magnitude
errors = np.abs(y_test - y_pred)  # Calculate prediction errors
plt.scatter(y_test, y_pred, c=errors, cmap='coolwarm', s=70, alpha=0.8, edgecolor='black', label='Predictions')

# Add a color bar to represent error magnitude
cbar = plt.colorbar()
cbar.set_label('Prediction Error', fontsize=12)

# Titles and labels
plt.title('Actual vs Predicted Values', fontsize=16, fontweight='bold')
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)

# Grid and legend
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12, loc='upper left')

# Adjust layout and display the plot
plt.tight_layout()

# Save the plot 
plt.savefig('./advanced_plots/actual_vs_predicted_values.png')
# Show the plot
plt.show()

#! Need the analysis of which factor or a combination of factors is affecting the salary the most, in the model above, trained using Linear Regression on multiple X - independent variables. 

# Extract feature names
feature_names = data.columns[:-1]  #! All columns except the target variable (Salary)

# Extract model coefficients using the coef_ attribute of the LinearRegression model. 
coefficients = model.coef_

# Create a DataFrame for visualization and sorting. Need to create a DataFrame to visualize the feature importance.
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

# Plot feature importance with annotations. 
plt.figure(figsize=(12, 8))
barplot = sns.barplot(
    x='Coefficient',
    y='Feature',
    data=importance_df,
    palette='coolwarm',
    orient='h',
    hue=None,
)

# This loop adds annotations to the bars in the barplot. 
for bar in barplot.patches:
    # Get the length of the bar (its coefficient value)
    width = bar.get_width()

    # Determine the position of the annotation:
    # - For positive coefficients, position slightly inside the bar's end (width - 0.1)
    # - For negative coefficients, position slightly inside the bar's start (width + 0.1)
    position = width - 0.1 if width > 0 else width + 0.1

    # Place the annotation (coefficient value) directly on the bar
    barplot.annotate(
        f'{width:.2f}',  # Format the coefficient value to two decimal places
        (position, bar.get_y() + bar.get_height() / 2),  # Position: centered vertically on the bar
        ha='right' if width > 0 else 'left',  # Align text to the right for positive, left for negative coefficients
        va='center',  # Vertically align text at the center of the bar
        fontsize=12,  # Use a professional and readable font size
        color='white',  # Use white text for high contrast against bar colors
        xytext=(0, 0),  # No offset applied for simplicity
        textcoords='offset points'  # Position relative to the bar itself
    )

# Customize plot for better layout and readability
plt.title('Feature Importance for Predicting Salary', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Coefficient Value (Impact)', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Apply tight layout to ensure everything fits well
plt.tight_layout()

# Save the plot with no clipping
plt.savefig('./advanced_plots/feature_importance_on_bar.png', bbox_inches='tight')
# Show the plot
plt.show()

# Save the model to a file using joblib.

joblib.dump(model, './model/salary_prediction_ML_regression_model.pkl')
print("Model saved successfully!")


"""
Model Summary:

1. Slope (Coefficients):
- These represent the effect of each feature on the salary.
- A positive coefficient means that as the feature value increases, the salary tends to increase.
- A negative coefficient means that as the feature value increases, the salary tends to decrease.

2. Intercept:
- The intercept (102029.02) is the baseline salary when all independent variables are zero.

3. Mean Squared Error (MSE):
- The MSE (918560764.39) measures the average squared difference between actual and predicted salaries.
- Lower values indicate better performance.

4. R² Score:
- The R² score (0.68) indicates that the model explains 68% of the variance in salary.
- The remaining 32% is due to factors not included in the model or randomness.

Feature Coefficients and Interpretations:
| Feature               | Coefficient  | Interpretation                                                                     |
|-----------------------|--------------|-----------------------------------------------------------------------------------|
| YearsOfExperience     | 8985.79      | A strong positive influence: Salary increases by $8,985.79 per additional year.  |
| Gender                | 6441.43      | A significant positive influence: Being a specific gender increases salary by $6,441.43. |
| EducationLevel        | 2082.74      | A moderate positive influence: Higher education levels increase salary by $2,082.74. |
| JobTitle              | -26.95       | A negligible negative influence: Slight decrease in salary by $26.95 per unit.    |
| Age                   | -1950.68     | A moderate negative influence: Salary decreases by $1,950.68 per unit of age.     |

Key Takeaways:
1. Years of Experience is the most influential factor, showing the highest positive impact on salary.
2. Gender shows a significant influence, likely indicating a gender-related salary disparity.
3. Education Level positively contributes to salary, but its impact is smaller compared to experience.
4. Age has a negative impact, potentially reflecting trends in seniority or other related factors.
"""
