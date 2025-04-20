# ui/mesa_app/model.py
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agents import DonorAgent, RecipientAgent, TransportAgent

class FoodModel(Model):
    def __init__(self, N_donors, N_recipients, N_transport, width, height):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)

        # Metrics
        self.delivered_count = 0
        self.wasted_count = 0

        # Donation store: pos -> list of food_types
        self._donations = {}

        # 1) Donors
        for i in range(N_donors):
            a = DonorAgent(f"D{i}", self, food_type="vegetables")
            self.schedule.add(a)
            self.grid.place_agent(a, self.random_cell())

        # 2) Recipients
        for i in range(N_recipients):
            a = RecipientAgent(f"R{i}", self)
            self.schedule.add(a)
            self.grid.place_agent(a, self.random_cell())

        # 3) Transporters
        for i in range(N_transport):
            a = TransportAgent(f"T{i}", self)
            self.schedule.add(a)
            self.grid.place_agent(a, self.random_cell())

        # DataCollector for live charts
        self.datacollector = DataCollector({
            "Delivered": lambda m: m.delivered_count,
            "Wasted":    lambda m: m.wasted_count,
        })

    def random_cell(self):
        return (self.random.randrange(self.grid.width),
                self.random.randrange(self.grid.height))

    def create_donation(self, food_type, pos):
        self._donations.setdefault(pos, []).append(food_type)

    def pop_donation_at(self, pos):
        bucket = self._donations.get(pos)
        if bucket:
            return bucket.pop(0)
        return None

    def find_nearest_donation(self, pos):
        if not self._donations:
            return pos
        return min(self._donations.keys(),
                   key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))

    def find_nearest_recipient(self, pos):
        recs = [ag.pos for ag in self.schedule.agents
                if isinstance(ag, RecipientAgent)]
        return min(recs,
                   key=lambda r: abs(r[0] - pos[0]) + abs(r[1] - pos[1]))

    def next_step(self, current, target):
        x, y = current
        tx, ty = target
        dx = (tx > x) - (tx < x)
        dy = (ty > y) - (ty < y)
        return (x + dx, y + dy)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

def agent_portrayal(agent):
    if isinstance(agent, DonorAgent):
        return {"Shape":"circle", "Color":"green",  "r":0.5, "Filled":True, "Layer":0}
    if isinstance(agent, TransportAgent):
        return {"Shape":"rect",   "Color":"blue",   "w":0.5, "h":0.5, "Filled":True, "Layer":1}
    if isinstance(agent, RecipientAgent):
        return {"Shape":"circle", "Color":"orange", "r":0.3, "Filled":True, "Layer":2}