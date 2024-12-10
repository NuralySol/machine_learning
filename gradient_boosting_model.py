
#! Gradient Boosting Model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the data set.
data = pd.read_csv('./data/salary_dataset.csv')

# Print the head of the data.
print(f"Head of the data:\n{data.head()}")
print("-" * 100)
# Print the shape of the data.
print(f"Shape of the data: {data.shape}")
print("-" * 100)
# Print the data types of the data.
print(f"Data types:\n{data.dtypes}")
print("-" * 100)
# Print the summary statistics of the data.
print(f"Summary statistics:\n{data.describe()}")
print("-" * 100)
# Check for missing values.
print(f"Missing values:\n{data.isnull().sum()}")
print("-" * 100)

# Drop the missing values, in the data set.
data = data.dropna()

# Check for missing values and drop them
print(f"Missing values: \n{data.isnull().sum()}")
data = data.dropna()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = labelencoder.fit_transform(data[column])

# Split the data into independent ('X') and dependent ('y') variables
X = data.iloc[:, :-1].values  # Features (all columns except Salary)
y = data.iloc[:, -1].values   # Target variable (Salary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Store results for comparison
results = []

# 1. Sklearn Gradient Boosting
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=0)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)
results.append({'Model': 'Sklearn Gradient Boosting', 'MSE': mse_gbr, 'R2': r2_gbr})

# 2. XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
results.append({'Model': 'XGBoost', 'MSE': mse_xgb, 'R2': r2_xgb})

# 3. LightGBM
lgb_model = LGBMRegressor(n_estimators=100, random_state=0)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)
results.append({'Model': 'LightGBM', 'MSE': mse_lgb, 'R2': r2_lgb})

# Compare Results
results_df = pd.DataFrame(results)

# Print results
print("\nModel Comparison:")
print(results_df)

# Importance DataFrame
feature_names = data.columns[:-1]  # All columns except Salary
feature_importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Visualize results: Model R² Comparison
plt.figure(figsize=(12, 8))

# Barplot for R² Scores
sns.barplot(x='R2', y='Model', data=results_df, palette='viridis', edgecolor='black')

# Add annotations for each bar
for bar in plt.gca().patches:
    plt.gca().text(
        bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
        f'{bar.get_width():.2f}',  # Format to two decimal places
        ha='left', va='center', fontsize=12, color='black', weight='bold'
    )

# Customize plot aesthetics
plt.title('Model R² Comparison', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('R² Score', fontsize=14)
plt.ylabel('Model', fontsize=14)
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
# Save the plot
plt.savefig('./random_forrest_&_gradient_boosting/model_comparison.png', bbox_inches='tight')
# Show the plot
plt.show()

# Feature Importance Visualization (XGBoost)
plt.figure(figsize=(12, 8))

# Barplot for feature importance
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm', edgecolor='black')

# Add annotations for each bar
for bar in plt.gca().patches:
    plt.gca().text(
        bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
        f'{bar.get_width():.2f}',  # Format to two decimal places
        ha='left', va='center', fontsize=12, color='black', weight='bold'
    )

# Customize plot aesthetics
plt.title('Feature Importance (XGBoost)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('./random_forrest_&_gradient_boosting/feature_importance_xgboost.png', bbox_inches='tight')

# Show the plot
plt.show()

# Save the best model
best_model = xgb_model
joblib.dump(best_model, './model/xgb_model.pkl')