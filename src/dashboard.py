import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import plotly.express as px
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div(
    [
        html.H1("Fraud Detection Dashboard"),
        # Summary Boxes
        html.Div(
            [
                html.Div(id="total-transactions", className="summary-box"),
                html.Div(id="fraud-cases", className="summary-box"),
                html.Div(id="fraud-percentage", className="summary-box"),
            ],
            style={"display": "flex", "justify-content": "space-around"},
        ),
        # Line Chart for Fraud Trends
        dcc.Graph(id="fraud-trends-chart"),
        # Bar Chart for Device Analysis
        dcc.Graph(id="device-analysis-chart"),
    ]
)


# Fetch data from Flask API
def fetch_fraud_insights():
    response = requests.get("http://localhost:5000/fraud-insights")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch fraud insights")


# Callbacks to update dashboard components
@app.callback(
    [
        Output("total-transactions", "children"),
        Output("fraud-cases", "children"),
        Output("fraud-percentage", "children"),
        Output("fraud-trends-chart", "figure"),
        Output("device-analysis-chart", "figure"),
    ],
    [Input("interval-component", "n_intervals")],
)
def update_dashboard(n):
    insights = fetch_fraud_insights()
    fraud_data = pd.read_csv("data/processed_fraud_data.csv")

    # Update summary boxes
    total_transactions = f"Total Transactions: {insights['total_transactions']}"
    fraud_cases = f"Fraud Cases: {insights['fraud_cases']}"
    fraud_percentage = f"Fraud Percentage: {insights['fraud_percentage']}%"

    # Update fraud trends chart
    fraud_trends = pd.Series(insights["fraud_trends"]).reset_index()
    fraud_trends.columns = ["Date", "Fraud Cases"]
    fraud_trends_chart = px.line(
        fraud_trends, x="Date", y="Fraud Cases", title="Fraud Trends Over Time"
    )

    # Update device analysis chart
    device_analysis = fraud_data.groupby("device_id")["class"].sum().reset_index()
    device_analysis_chart = px.bar(
        device_analysis, x="device_id", y="class", title="Fraud Cases by Device"
    )

    return (
        total_transactions,
        fraud_cases,
        fraud_percentage,
        fraud_trends_chart,
        device_analysis_chart,
    )


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
