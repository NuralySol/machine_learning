# Dash Project visualization for the data. Main app.py root file.
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib

# Initialize Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Model Visualization Dashboard"

# Define the plots and summaries
plots = {
    "Salary Distribution": "salary_distribution.png",
    "Correlation Matrix": "correlation_matrix.png",
    "Pairplot": "pairplot.png",
    "Feature Importance": "feature_importance.png",
    "Feature Importance on Bar": "feature_importance_on_bar.png",
    "Feature Importance (XGBoost)": "feature_importance_xgboost.png",
    "Model Comparison": "model_comparison.png",
    "Salary Prediction Plot": "salary_prediction_plot.png",
    "Actual vs Predicted (Subset)": "actual_vs_predicted_subset.png",
    "Actual vs Predicted Values": "actual_vs_predicted_values.png",
    "Residuals Plot": "residuals_plot.png",
}

summaries = {
    "Salary Distribution": "This histogram shows the distribution of salary values in the dataset, with a kernel density estimate overlay.",
    "Correlation Matrix": "The correlation matrix displays the relationships between different features in the dataset. Values close to 1 indicate a strong positive correlation.",
    "Pairplot": "The pairplot shows pairwise relationships between features, with distributions and scatter plots for comparison.",
    "Feature Importance": "This bar chart visualizes the importance of each feature in predicting salaries using the Random Forest model.",
    "Feature Importance on Bar": "This plot provides feature importance annotated directly on the bars for better interpretability.",
    "Feature Importance (XGBoost)": "Feature importance as calculated by the XGBoost model, highlighting the most influential variables in predicting salaries.",
    "Model Comparison": "This chart compares the performance of different models using their R² scores.",
    "Salary Prediction Plot": "This regression plot visualizes the relationship between years of experience and salary, with predictions shown as a trend line.",
    "Actual vs Predicted (Subset)": "This plot compares actual vs. predicted salary values for a subset of test samples. Blue points represent actual values, while red points show predictions.",
    "Actual vs Predicted Values": "This plot shows the overall comparison of actual vs. predicted salaries across all test samples.",
    "Residuals Plot": "The residuals plot displays the differences between actual and predicted salaries. A balanced distribution around 0 indicates a good fit.",
}

# Load the trained model (e.g., Random Forest)
model = joblib.load('./model/salary_prediction_model.pkl')

# Define layouts
home_layout = html.Div(
    children=[
        html.H1("Model Visualization Dashboard", style={"text-align": "center"}),

        # Dropdown for selecting plots
        html.Div(
            children=[
                html.Label("Select Plot to View:"),
                dcc.Dropdown(
                    id="plot-selector",
                    options=[{"label": name, "value": name} for name in plots.keys()],
                    value=list(plots.keys())[0],  
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

all_plots_layout = html.Div(
    children=[
        html.H1("All Plots", style={"text-align": "center", "margin-bottom": "20px"}),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Img(
                            src=app.get_asset_url(path),
                            style={
                                "width": "100%",
                                "border": "1px solid #ddd",
                                "border-radius": "10px",
                                "margin-bottom": "20px",
                            },
                        ),
                        html.P(summary, style={"text-align": "center", "font-size": "14px"}),
                    ]
                )
                for name, path, summary in zip(plots.keys(), plots.values(), summaries.values())
            ],
            style={"max-width": "800px", "margin": "0 auto"},
        ),
    ]
)

# Define main layout with routing
app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(
            children=[
                dcc.Link("Home", href="/", style={"margin-right": "15px", "font-size": "18px"}),
                dcc.Link("All Plots", href="/all-plots", style={"font-size": "18px"}),
            ],
            style={"text-align": "center", "margin-bottom": "20px"},
        ),
        html.Div(id="page-content"),
    ]
)

# Callback for dynamic page routing
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/all-plots":
        return all_plots_layout
    return home_layout

# Callback for updating the plot and summary based on user selection.
@app.callback(
    [Output("plot-display", "children"), Output("summary-display", "children")],
    [Input("plot-selector", "value")],
)
def update_plot_and_summary(selected_plot):
    if selected_plot is None:
        # Fallback to the first plot if no selection is made
        selected_plot = list(plots.keys())[0]
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