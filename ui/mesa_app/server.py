from mesa.visualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from model import FoodModel, agent_portrayal

if __name__ == "__main__":
    # Define the grid visualization
    canvas_element = CanvasGrid(agent_portrayal, grid_width=20, grid_height=20, canvas_width=500, canvas_height=500)

    # Define chart elements for metrics
    chart_delivered = ChartModule([{"Label": "Delivered", "Color": "Blue"}])
    chart_wasted = ChartModule([{"Label": "Wasted", "Color": "Red"}])

    # Launch the ModularServer
    server = ModularServer(
        FoodModel,
        [canvas_element, chart_delivered, chart_wasted],
        "Food Redistribution Simulation",
        {"N_donors": 10, "N_recipients": 5, "N_transport": 3, "width": 20, "height": 20}
    )
    server.port = 8521
    server.launch()