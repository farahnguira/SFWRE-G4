import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os
from datetime import datetime

app = dash.Dash(__name__)

# Load data from outputs directory
def load_data():
    outputs_dir = "SFWRE-G4\amine\data\tarji"

    # Expected files based on the provided list
    scenarios = ["disaster", "spike", "urban"]
    capacities = [50, 100, 200, 400, 600]
    days = range(1, 31)

    # Load summary files (primary data source)
    # Updated to match the actual file naming: summary_{scenario}_{capacity}_{day}.csv
    summary_files = [f"summary_{scenario}_{capacity}_{day}.csv" for scenario in scenarios for capacity in capacities for day in days]
    existing_summary_files = []
    summary_dfs = []
    
    # Debug: Check if files exist
    print(f"Looking for summary files in directory: {os.path.abspath(outputs_dir)}")
    for file in summary_files:
        file_path = os.path.join(outputs_dir, file)
        if os.path.exists(file_path):
            existing_summary_files.append(file)
        else:
            print(f"File not found: {file_path}")

    print(f"Found {len(existing_summary_files)} summary files: {existing_summary_files}")
    
    # Load summary files
    for file in existing_summary_files:
        try:
            file_path = os.path.join(outputs_dir, file)
            df = pd.read_csv(file_path)
            print(f"Successfully loaded: {file}")
            
            # Extract scenario, capacity, and day from filename
            parts = file.split("_")
            scenario = parts[1]
            capacity = int(parts[2])
            day = int(parts[3].split(".")[0])
            df["scenario"] = scenario
            df["capacity"] = capacity
            df["day"] = day
            
            # Use existing timestamp if present, otherwise generate based on day
            if "timestamp" not in df.columns or pd.isna(df["timestamp"].iloc[0]):
                df["timestamp"] = pd.to_datetime("2025-05-06") + pd.to_timedelta(day - 1, unit="d")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Strip the time portion, keep only the date
            df["timestamp"] = df["timestamp"].dt.date
            summary_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")

    if summary_dfs:
        df = pd.concat(summary_dfs, ignore_index=True)
        print("Loaded summary data:")
        print("Columns:", df.columns.tolist())
        print("Summary of food_wasted:")
        print(df["food_wasted"].describe())
        print("Rows per scenario/capacity combination:")
        print(df.groupby(["scenario", "capacity"]).size())
    else:
        # Fallback to allocated items files
        # Updated to match the actual file naming: allocated_items_{scenario}_{capacity}.csv
        allocated_files = [f"allocated_items_{scenario}_{capacity}.csv" for scenario in scenarios for capacity in capacities]
        existing_allocated_files = []
        
        # Debug: Check if allocated files exist
        print(f"Looking for allocated items files in directory: {os.path.abspath(outputs_dir)}")
        for file in allocated_files:
            file_path = os.path.join(outputs_dir, file)
            if os.path.exists(file_path):
                existing_allocated_files.append(file)
            else:
                print(f"File not found: {file_path}")

        print(f"Found {len(existing_allocated_files)} allocated items files: {existing_allocated_files}")
        
        allocated_dfs = []
        for file in existing_allocated_files:
            try:
                file_path = os.path.join(outputs_dir, file)
                df = pd.read_csv(file_path)
                print(f"Successfully loaded: {file}")
                
                parts = file.split("_")
                scenario = parts[2]
                capacity = int(parts[3].split(".")[0])
                df["scenario"] = scenario
                df["capacity"] = capacity
                allocated_dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

        if allocated_dfs:
            df = pd.concat(allocated_dfs, ignore_index=True)
            daily_agg = df.groupby(["day", "scenario", "capacity"]).agg({
                "quantity": "sum",
                "adjusted_priority": "sum"
            }).reset_index()
            daily_agg = daily_agg.rename(columns={"quantity": "total_quantity_delivered", "adjusted_priority": "total_priority_score"})
            daily_agg["food_wasted"] = 0
            daily_agg["timestamp"] = pd.to_datetime("2025-05-06") + pd.to_timedelta(daily_agg["day"] - 1, unit="d")
            # Strip the time portion for fallback data
            daily_agg["timestamp"] = daily_agg["timestamp"].dt.date
            daily_agg["algorithm"] = "dp_knapsack"
            food_types = df["type"].unique()
            for food_type in food_types:
                daily_agg[f"quantity_{food_type}"] = df[df["type"] == food_type].groupby(["day", "scenario", "capacity"])["quantity"].sum().reindex(
                    daily_agg.index, fill_value=0
                )
            df = daily_agg
            print("Aggregated allocated items data:")
        else:
            print("No summary or allocated items files found in tarji directory. Using fallback data.")
            df = pd.DataFrame({
                "timestamp": [(pd.to_datetime("2025-05-06") + pd.to_timedelta(i, unit="d")).date() for i in range(4)],
                "scenario": ["urban"] * 4,
                "algorithm": ["dp_knapsack"] * 4,
                "capacity": [50] * 4,
                "food_wasted": [2, 5, 3, 7],
                "total_quantity_delivered": [50, 48, 45, 40],
                "quantity_bakery": [10, 9, 8, 7],
                "quantity_canned": [5, 5, 4, 4],
                "quantity_dairy": [15, 14, 13, 12],
                "quantity_dry goods": [5, 5, 4, 4],
                "quantity_meat": [10, 9, 8, 7],
                "quantity_vegetables": [20, 19, 18, 17]
            })

    print("Loaded DataFrame:")
    print(df[["timestamp", "scenario", "capacity", "food_wasted", "total_quantity_delivered"]])
    return df

df = load_data()

# Transform food type quantities to long format
food_types = ["bakery", "canned", "dairy", "dry goods", "meat", "vegetables"]
df_long = pd.melt(
    df,
    id_vars=["timestamp", "scenario", "algorithm", "capacity", "food_wasted", "total_quantity_delivered"],
    value_vars=[f"quantity_{ft}" for ft in food_types],
    var_name="food_type",
    value_name="quantity"
)
df_long["food_type"] = df_long["food_type"].str.replace("quantity_", "").str.capitalize()

# Ensure data types
df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert back to datetime for plotting
df["food_wasted"] = pd.to_numeric(df["food_wasted"], errors="coerce")
df["total_quantity_delivered"] = pd.to_numeric(df["total_quantity_delivered"], errors="coerce")
df["scenario"] = df["scenario"].astype(str)
df["algorithm"] = df["algorithm"].astype(str)
df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
df_long["scenario"] = df_long["scenario"].astype(str)
df_long["algorithm"] = df_long["algorithm"].astype(str)
df_long["capacity"] = pd.to_numeric(df_long["capacity"], errors="coerce")

# Layout
app.layout = html.Div(className="bg-gray-100 p-4", children=[
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label="Charts", children=[
            html.Div(className="flex flex-col space-y-4 p-4", children=[
                dcc.Dropdown(id="scenario-dropdown", options=[{"label": s, "value": s} for s in df["scenario"].unique()], value=df["scenario"].iloc[0]),
                dcc.Dropdown(id="algorithm-dropdown", options=[{"label": a, "value": a} for a in df["algorithm"].unique()], value=df["algorithm"].iloc[0]),
                # Fixed the syntax error: replaced 'c' with 's' in the list comprehension
                dcc.Dropdown(id="capacity-dropdown", options=[{"label": str(s), "value": s} for s in sorted(df["capacity"].unique())], value=df["capacity"].iloc[0]),
                html.Button("Run Simulation", id="run-simulation-btn", className="bg-blue-500 text-white p-2 rounded mt-2"),
                dcc.Graph(id="waste-line-chart"),
                dcc.Graph(id="delivered-bar-chart")
            ])
        ]),
        dcc.Tab(label="KPI Dashboard", children=[
            html.Div(className="flex flex-col space-y-4 p-4", children=[
                dcc.Dropdown(id="scenario-dropdown-kpi", options=[{"label": s, "value": s} for s in df["scenario"].unique()], value=df["scenario"].iloc[0]),
                dcc.Dropdown(id="algorithm-dropdown-kpi", options=[{"label": a, "value": a} for a in df["algorithm"].unique()], value=df["algorithm"].iloc[0]),
                dcc.Dropdown(id="capacity-dropdown-kpi", options=[{"label": str(c), "value": c} for c in sorted(df["capacity"].unique())], value=df["capacity"].iloc[0]),
                html.Div(id="kpi-boxes", className="grid grid-cols-1 md:grid-cols-3 gap-4")
            ])
        ])
    ])
])

# Callback for charts
@app.callback(
    [Output("waste-line-chart", "figure"), Output("delivered-bar-chart", "figure")],
    [Input("run-simulation-btn", "n_clicks")],
    [State("scenario-dropdown", "value"), State("algorithm-dropdown", "value"), State("capacity-dropdown", "value")]
)
def update_charts(n_clicks, scenario, algorithm, capacity):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    filtered_df = df[(df["scenario"] == scenario) & (df["algorithm"] == algorithm) & (df["capacity"] == capacity)]
    print(f"Filtered DataFrame (scenario={scenario}, algorithm={algorithm}, capacity={capacity}):")
    print(filtered_df[["timestamp", "food_wasted", "total_quantity_delivered"]])
    filtered_df_long = df_long[(df_long["scenario"] == scenario) & (df_long["algorithm"] == algorithm) & (df_long["capacity"] == capacity)]
    print(f"Filtered df_long for bar chart:")
    print(filtered_df_long[["food_type", "quantity"]])
    line_fig = px.line(filtered_df, x="timestamp", y="food_wasted", title="Food Waste Over Time")
    if len(filtered_df) <= 1:
        line_fig = px.scatter(filtered_df, x="timestamp", y="food_wasted", title="Food Waste Over Time")
        line_fig.update_traces(marker=dict(size=12))
    # Format x-axis to show only dates, no time
    line_fig.update_xaxes(
        tickformat="%Y-%m-%d",
        tickangle=45
    )
    bar_fig = px.bar(filtered_df_long.groupby("food_type")["quantity"].sum().reset_index(),
                     x="food_type", y="quantity", title="Delivered by Food Type")
    return line_fig, bar_fig

# Callback for KPIs
@app.callback(
    Output("kpi-boxes", "children"),
    [Input("scenario-dropdown-kpi", "value"), Input("algorithm-dropdown-kpi", "value"), Input("capacity-dropdown-kpi", "value")]
)
def update_kpis(scenario, algorithm, capacity):
    filtered_df = df[(df["scenario"] == scenario) & (df["algorithm"] == algorithm) & (df["capacity"] == capacity)]
    baseline_waste = 20
    waste_reduction = (1 - filtered_df["food_wasted"].mean() / baseline_waste) * 100 if not filtered_df.empty else 0
    route_efficiency = filtered_df["total_quantity_delivered"].sum() / len(filtered_df) if not filtered_df.empty else 0
    kpi_boxes = [
        html.Div(className="bg-blue-100 p-4 rounded-lg shadow", children=[
            html.H3("Waste Reduction", className="text-blue-600"),
            html.P(f"{waste_reduction:.2f}%")
        ]),
        html.Div(className="bg-blue-100 p-4 rounded-lg shadow", children=[
            html.H3("Route Efficiency", className="text-blue-600"),
            html.P(f"{route_efficiency:.2f} items/delivery")
        ]),
        html.Div(className="bg-blue-100 p-4 rounded-lg shadow", children=[
            html.H3("Prediction Accuracy", className="text-blue-600"),
            html.P("0.00%")
        ])
    ]
    return kpi_boxes

if __name__ == "__main__":
    app.run(debug=True, port=8050)