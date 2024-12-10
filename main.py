# Need the following libraries to be installed: pandas, numpy, matplotlib, seaborn, sklearn
# Need the libraries to perform the following tasks:
# 1. Import the dataset
# 2. Perform EDA
# 3. Perform Data Preprocessing
# 4. Split the dataset into training and testing
# 5. Train the model
# 6. Test the model
# 7. Evaluate the model
# 8. Make predictions
# 9. Visualize the predictions
# 10. Save the model
# 11. Load the model
# 12. Make predictions using the loaded model
# 13. Visualize the predictions using the loaded model

#! Pandas is used for data manipulation and analysis, it is a fast, powerful, flexible and easy to use open-source data analysis and data manipulation library built on top of the Python programming language.
#! NumPy is used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.
#! Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
#! Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
#! Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

import pandas as pd                   
import numpy as np                   
import matplotlib.pyplot as plt       
import seaborn as sns                
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Importing the dataset
data = pd.read_csv('./data/salary_dataset.csv')
print(f"Head of the imported data is : \n{data.head()}") # Displays the first 5 rows of the dataset
print("-" * 100)
print(f"Describe the imported data is : \n {data.describe()}") # Displays the statistical summary of the dataset
print("-" * 100)
print(f"Information of the imported data is : \n {data.info()}") # Displays the information of the dataset
print("-" * 100)
print(f"Shape of the imported data is : \n {data.shape}") # Displays the shape of the dataset. Shape is a tuple that gives dimensions of the data.
print("-" * 100)
print(f"Columns of the imported data is : \n {data.columns}") # Displays the columns of the dataset
print("-" * 100)
print(f"Checking for null values in the imported data : \n {data.isnull().sum()}") # Displays the sum of null values in the dataset. If the sum is 0, then there are no null values in the dataset. 
print("-" * 100)

# Clean the data. Remove the null values from the dataset. 
data = data.dropna()
print(f"Checking for the null values after removing them :", data.isnull().sum())
print("-" * 100)

# Perform EDA (Exploratory Data Analysis). EDA is an approach to analyzing data sets to summarize their main characteristics, often with visual methods.
# Plotting the distribution of the target variable 'Salary' using a histogram. A histogram is a graphical representation of the distribution of a dataset. It shows the frequency of values in a dataset.
plt.figure(figsize=(12, 8))
sns.histplot(data['Salary'], kde=True, color='royalblue', bins=30, edgecolor='black')
plt.title('Distribution of Salary', fontsize=16, fontweight='bold')
plt.xlabel('Salary', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig('./plots/salary_distribution.png')
plt.show()

# Plotting the pairplot to see the relationships between variables. A pairplot is a grid of plots that shows the relationships between pairs of variables in a dataset. It shows the relationship between each numerical variable in the dataset.
# Define a function to calculate and annotate correlation coefficients. The annotate_corr function calculates the correlation coefficient between two variables and annotates it on the scatter plot.
def annotate_corr(x, y, **kwargs):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.text(
        0.05, 0.95,
        f'r = {r:.2f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

# Enhanced Pairplot with Hue and Annotations
pairgrid = sns.pairplot(
    data,
    diag_kind='kde',
    hue='Gender',  # Replace with a relevant categorical column
    palette='husl',
    corner=True,
    markers=['o', 's']
)

# Add annotations to the scatter plots
pairgrid.map_lower(annotate_corr)

# Adjust layout and aesthetics
plt.suptitle('Pairplot of the Dataset with Annotations', y=1.02, fontsize=20, fontweight='bold')

# Use `rect` to add space for the title
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusts the top margin for the title

# Save and show the plot
plt.savefig('./plots/pairplot.png', bbox_inches='tight')
plt.show()

"""
# Plotting the correlation matrix. A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables.
# Select only numerical columns. Need to convert the categorical columns to numerical columns.
# The correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. Closer to 1 means strong positive correlation, closer to -1 means strong negative correlation, and closer to 0 means no correlation.
"""

# Make sure that data is in numerical format. Select only numerical columns. Need to convert the categorical columns to numerical columns.
numerical_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(numerical_data.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('./plots/correlation_matrix.png')  
plt.show()

# Split the dataset into training and testing models. The training set is used to train the model and the testing set is used to test the model.
# The independent variable is 'YearsExperience' and the dependent variable is 'Salary'. Big X is the independent variable and small y is the dependent variable. We need to predict the salary based on the years of experience.
X = data[['YearsOfExperience']]
y = data['Salary']

#! Split the dataset into training and testing sets with 80% training and 20% testing. The standard split ratio is 80% training and 20% testing. (random_state=42) is used to reproduce the same results. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model. Linear Regression is used to train the model. Linear regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables. 

model = LinearRegression()
model.fit(X_train, y_train)

# Test the model. Predict the salary based on the years of experience.
y_pred = model.predict(X_test)

# Evaluate the model. Calculate the mean squared error and r2 score. The mean squared error is the average of the squares of the errors. The r2 score is the proportion of the variance in the dependent variable that is predictable from the independent variable.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the mean squared error and r2 score. 
print(f"Mean Squared Error : {mse:.2f}")
print(f"R2 Score : {r2:.2f}")
print("-" * 100)

"""
Interpretation of Model Performance:

1. Mean Squared Error (MSE): 982050061.92
- The MSE measures the average squared difference between the actual 
and predicted values. 
- A higher MSE suggests larger prediction errors. However, the 
magnitude of the MSE should be evaluated relative to the scale of 
the dependent variable (Salary). 
- If salaries are in the tens or hundreds of thousands, a large MSE 
might still be reasonable.

2. R² Score: 0.66
- The R² score indicates that the model explains 66% of the variance 
in Salary based on Years of Experience.
- This reflects a moderately strong relationship between the variables.
- The remaining 34% of the variance is due to factors not captured by 
the model, such as other features like Education Level or Job Title.

Overall, the model performs moderately well but could be improved by:
- Adding more features to better explain the variability in Salary.
- Exploring non-linear relationships between Years of Experience and Salary.
- Refining the data by removing outliers or handling anomalies.
"""

# Improved Visualization of Predictions
plt.figure(figsize=(12, 8))

# Scatter plot for actual data points
plt.scatter(X_test, y_test, color='skyblue', edgecolor='black', s=80, label='Actual Data')

# Regression line for predicted values
plt.plot(X_test, y_pred, color='darkorange', linewidth=2.5, label='Regression Line')

# Enhancing the aesthetics
plt.title('Salary Prediction vs. Years of Experience', fontsize=16, fontweight='bold')
plt.xlabel('Years of Experience', fontsize=14)
plt.ylabel('Salary', fontsize=14)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('./plots/salary_prediction_plot.png')  

# Show the plot
plt.show()

# Print the prediction for a specific value of years of experience. With user input, the model can predict the salary based on the years of experience.
years_of_experience = user_input = float(input("Enter the years of experience to predict the salary: "))
predicted_salary = model.predict([[years_of_experience]]) # two square brackets are used to pass a 2D array to the predict function.
print(f"The predicted salary for {years_of_experience:.1f} years of experience is: ${predicted_salary[0]:,.2f}")
print(f"The accuracy of prediction is: {r2:.2f} (R² Score)") 
#! R^2 Score of (0.9) or above is considered an excellent R² score, 0.7-0.9 is considered good, 0.5-0.7 is considered moderate, and below 0.5 is considered weak.
print("-" * 100)

# Save the model. The model is saved as a .pkl file using the joblib library. The model can be loaded and used for predictions in the future. Even though it is a moderate predictor, it can still be used for the future predictions.
joblib.dump(model, './model/salary_prediction_model.pkl')
print("Model Saved Successfully!")

# Load the model. The saved model can be loaded using the joblib library. The loaded model can be used to make predictions. The model is saved in loaded_model variable.
loaded_model = joblib.load('./model/salary_prediction_model.pkl')
print("Model Loaded Successfully!")

