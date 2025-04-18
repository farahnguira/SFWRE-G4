from mesa import Model, Agent 
from mesa.time import RandomActivation 
from mesa.space import MultiGrid 
import random 
 
class FoodModel(Model): 
    def __init__(self, N, width, height): 
        super().__init__() 
        self.schedule = RandomActivation(self) 
        self.grid = MultiGrid(width, height, torus=True)  # Add a grid 
        for i in range(N): 
            a = Agent(i, self) 
            self.schedule.add(a) 
            # Place agent randomly on the grid 
            x = random.randrange(width) 
            y = random.randrange(height) 
            self.grid.place_agent(a, (x, y)) 
 
    def step(self): 
        self.schedule.step() 
 
def agent_portrayal(agent): 
    return {"Shape": "circle", "Color": "red", "r": 0.5} 
