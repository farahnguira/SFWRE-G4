# server.py
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from MesaApp import FoodDistributionModel
import csv
import json

# Portrayal function for visualization
def agent_portrayal(agent):
    from composants import TransportAgent
    if isinstance(agent, TransportAgent):
        portrayal = {
            "Shape": "circle",
            "Filled": "true",
            "r": 0.5,
            "Layer": 1
        }
        # Color by status
        status = getattr(agent, "status", "pending")
        portrayal["Color"] = "green" if status == "delivered" else "blue"
        # Label with item_type initial
        item = getattr(agent, "item_type", "")
        portrayal["text"] = item[0].upper() if item else ""
        portrayal["text_color"] = "white"
        return portrayal
    return None

# Utility to read solver logs (CSV or JSON)
def load_solver_log(path):
    data = []
    if path.endswith('.csv'):
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append({
                    'agent_id': int(row['agent_id']),
                    'x': int(row['x']),
                    'y': int(row['y']),
                    'item_type': row['item_type'],
                    'status': row.get('status', 'pending')
                })
    elif path.endswith('.json'):
        with open(path) as jsonfile:
            entries = json.load(jsonfile)
            for entry in entries:
                data.append({
                    'agent_id': int(entry['agent_id']),
                    'x': int(entry['x']),
                    'y': int(entry['y']),
                    'item_type': entry['item_type'],
                    'status': entry.get('status', 'pending')
                })
    return data

# Path to your solver log file (CSV or JSON)
log_path = "solver_log.json"
solver_log = load_solver_log(log_path)

# Model parameters
model_params = {
    'donators': None,        # replace with your donators list
    'recipients': None,      # replace with your recipients list
    'inventory': None,       # replace with your InventoryAgent instance
    'solver_log': solver_log
}

# Create the CanvasGrid for live visualization
grid = CanvasGrid(agent_portrayal, 50, 50, 600, 600)

# Launch the Mesa server
def launch_server():
    server = ModularServer(
        FoodDistributionModel,
        [grid],
        "Food Distribution Model",
        model_params
    )
    server.port = 8521
    server.launch()

if __name__ == '__main__':
    launch_server()
