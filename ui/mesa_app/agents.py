from mesa import Agent

class DonorAgent(Agent):
    """Each tick, makes a new donation at its position."""
    def __init__(self, unique_id, model, food_type):
        super().__init__(unique_id, model)
        self.food_type = food_type

    def step(self):
        # register a donation in the model
        self.model.create_donation(self.food_type, self.pos)


class RecipientAgent(Agent):
    """Each tick, picks up any donation sitting at its position."""
    def step(self):
        item = self.model.pop_donation_at(self.pos)
        if item:
            self.model.delivered_count += 1


class TransportAgent(Agent):
    """Shuttles one item at a time from nearest donor to nearest recipient."""
    def __init__(self, unique_id, model, capacity=1):
        super().__init__(unique_id, model)
        self.capacity = capacity
        self.cargo = None

    def step(self):
        if self.cargo is None:
            # move toward and pick up nearest donation
            target = self.model.find_nearest_donation(self.pos)
            self.model.grid.move_agent(self, self.model.next_step(self.pos, target))
            if self.pos == target:
                self.cargo = self.model.pop_donation_at(self.pos)
        else:
            # move toward and deliver to nearest recipient
            target = self.model.find_nearest_recipient(self.pos)
            self.model.grid.move_agent(self, self.model.next_step(self.pos, target))
            if self.pos == target:
                self.model.delivered_count += 1
                self.cargo = None
from mesa import Agent

class DonorAgent(Agent):
    """Each tick, makes a new donation at its position."""
    def __init__(self, unique_id, model, food_type):
        super().__init__(unique_id, model)
        self.food_type = food_type

    def step(self):
        # register a donation in the model
        self.model.create_donation(self.food_type, self.pos)


class RecipientAgent(Agent):
    """Each tick, picks up any donation sitting at its position."""
    def step(self):
        item = self.model.pop_donation_at(self.pos)
        if item:
            self.model.delivered_count += 1


class TransportAgent(Agent):
    """Shuttles one item at a time from nearest donor to nearest recipient."""
    def __init__(self, unique_id, model, capacity=1):
        super().__init__(unique_id, model)
        self.capacity = capacity
        self.cargo = None

    def step(self):
        if self.cargo is None:
            # move toward and pick up nearest donation
            target = self.model.find_nearest_donation(self.pos)
            self.model.grid.move_agent(self, self.model.next_step(self.pos, target))
            if self.pos == target:
                self.cargo = self.model.pop_donation_at(self.pos)
        else:
            # move toward and deliver to nearest recipient
            target = self.model.find_nearest_recipient(self.pos)
            self.model.grid.move_agent(self, self.model.next_step(self.pos, target))
            if self.pos == target:
                self.model.delivered_count += 1
                self.cargo = None
