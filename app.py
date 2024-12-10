# Dash Project visualization for the data. Main app.py root file.
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib

# Initialize Dash App
app = dash.Dash(__name__)

# Define the available images and their descriptions
plots = {
    "Actual vs Predicted (Subset)": "actual_vs_predicted_subset.png",
    "Actual vs Predicted Values": "actual_vs_predicted_values.png",
    "Correlation Matrix": "correlation_matrix.png",
    "Feature Importance": "feature_importance.png",
    "Feature Importance on Bar": "feature_importance_on_bar.png",
    "Feature Importance (XGBoost)": "feature_importance_xgboost.png",
    "Model Comparison": "model_comparison.png",
    "Pairplot": "pairplot.png",
    "Residuals Plot": "residuals_plot.png",
    "Salary Distribution": "salary_distribution.png",
    "Salary Prediction Plot": "salary_prediction_plot.png",
}

# Define summaries for the plots
summaries = {
    "Actual vs Predicted (Subset)": "This plot compares actual vs. predicted salary values for a subset of test samples. Blue points represent actual values, while red points show predictions.",
    "Actual vs Predicted Values": "This plot shows the overall comparison of actual vs. predicted salaries across all test samples.",
    "Correlation Matrix": "The correlation matrix displays the relationships between different features in the dataset. Values close to 1 indicate a strong positive correlation.",
    "Feature Importance": "This bar chart visualizes the importance of each feature in predicting salaries using the Random Forest model.",
    "Feature Importance on Bar": "This plot provides feature importance annotated directly on the bars for better interpretability.",
    "Feature Importance (XGBoost)": "Feature importance as calculated by the XGBoost model, highlighting the most influential variables in predicting salaries.",
    "Model Comparison": "This chart compares the performance of different models using their R² scores.",
    "Pairplot": "The pairplot shows pairwise relationships between features, with distributions and scatter plots for comparison.",
    "Residuals Plot": "The residuals plot displays the differences between actual and predicted salaries. A balanced distribution around 0 indicates a good fit.",
    "Salary Distribution": "This histogram shows the distribution of salary values in the dataset, with a kernel density estimate overlay.",
    "Salary Prediction Plot": "This regression plot visualizes the relationship between years of experience and salary, with predictions shown as a trend line.",
}

# Can load a different mode if needed to test the user interface.
# Load the trained model (e.g., Random Forest)
model = joblib.load('./model/salary_prediction_model.pkl')

# Layout of the Dash app
app.layout = html.Div(
    children=[
        html.H1("Model Visualization Dashboard", style={"text-align": "center"}),

        # Dropdown for selecting plots
        html.Div(
            children=[
                html.Label("Select Plot to View:"),
                dcc.Dropdown(
                    id="plot-selector",
                    options=[{"label": name, "value": name} for name in plots.keys()],
                    value=list(plots.keys())[0],  # Default to the first plot
                    style={"width": "50%"},
                ),
            ],
            style={"margin-bottom": "20px"},
        ),

        # Div to display the selected image and summary
        html.Div(id="plot-display"),
        html.Div(id="summary-display", style={"margin-top": "20px", "font-size": "16px"}),

        # User input for salary prediction
        html.Div(
            children=[
                html.H3("Predict Salary Based on Years of Experience"),
                html.Label("Enter Years of Experience:"),
                dcc.Input(
                    id="years-experience-input",
                    type="number",
                    placeholder="Enter years of experience (e.g., 5.5)",
                    style={"width": "30%", "margin-bottom": "10px"},
                ),
                html.Button("Predict", id="predict-button", n_clicks=0, style={"margin-left": "10px"}),
                html.Div(
                    id="prediction-output",
                    style={
                        "margin-top": "20px",
                        "font-size": "18px",
                        "font-weight": "bold",
                        "text-align": "center",
                    },
                ),
            ],
            style={"margin-top": "40px"},
        ),
    ]
)

# Callback to update the displayed image and summary based on dropdown selection
@app.callback(
    [Output("plot-display", "children"), Output("summary-display", "children")],
    [Input("plot-selector", "value")],
)
def update_plot_and_summary(selected_plot):
    plot_img = html.Img(
        src=app.get_asset_url(plots[selected_plot]),
        style={"width": "80%", "margin": "0 auto", "display": "block"},
    )
    plot_summary = html.P(summaries[selected_plot], style={"text-align": "center"})
    return plot_img, plot_summary


# Callback for predicting salary based on years of experience
@app.callback(
    Output("prediction-output", "children"),
    [Input("predict-button", "n_clicks")],
    [State("years-experience-input", "value")],
)
def predict_salary(n_clicks, years_of_experience):
    if n_clicks > 0:
        if years_of_experience is not None:
            # Predict salary using the model
            prediction = model.predict([[years_of_experience]])[0]
            return f"Predicted Salary: ${prediction:,.2f}"
        else:
            return "Please enter a valid number for years of experience."
    return ""

# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)