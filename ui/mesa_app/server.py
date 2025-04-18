from mesa.visualization import ModularServer 
from mesa.visualization.modules import CanvasGrid 
from model import FoodModel, agent_portrayal 
 
# Define the visualization grid 
grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500) 
 
server = ModularServer( 
    FoodModel, 
    [grid],  # Use CanvasGrid for visualization 
    "Food Redistribution Model", 
    {"N": 50, "width": 20, "height": 20} 
) 
 
if __name__ == "__main__": 
    server.port = 8521  # Specify a port to avoid conflicts 
    server.launch() 
