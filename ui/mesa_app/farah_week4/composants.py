from typing import Dict, List

from datetime import datetime

from food_item import FoodItem  # Importing FoodItem class


class TransportAgent(Agent):
    """Agent de transport autonome compatible Mesa."""

    def __init__(self, unique_id, model, pos=(0, 0), capacity=1):
        super().__init__(unique_id, model)
        self.pos = pos
        self.capacity = capacity
        self.cargo = None
        self.mission = None
        self.target = None

    def step(self):
        """Avance d’un pas dans la mission actuelle ou en attribue une nouvelle."""
        if self.mission is None:
            self.assign_mission()
        else:
            self.execute_mission()
    def add_collection_mission(self, food_item, source_donator, destination_inventory):
        """Ajoute une mission de collecte depuis un donateur vers l'inventaire."""
        if self.mission is None:
            self.mission = {
                'type': 'collect',
                'food': food_item.name,
                'source': source_donator,
                'destination': destination_inventory,
                'quantity': 1
            }
            self.target = source_donator.pos
            self.cargo = None

    def assign_mission(self):
        """Attribue une mission de collecte ou de livraison."""
        for donator in self.model.donators:
            for food_name, quantity in donator.donations.items():
                if quantity > 0:
                    self.mission = {
                        'type': 'collect',
                        'food': food_name,
                        'source': donator,
                        'destination': self.model.inventory,
                        'quantity': 1
                    }
                    self.target = donator.pos
                    return

        for recipient in self.model.recipients:
            for food_name, quantity in recipient.demand_profile.items():
                if quantity > 0 and self.model.inventory.current_load > 0:
                    self.mission = {
                        'type': 'deliver',
                        'food': food_name,
                        'source': self.model.inventory,
                        'destination': recipient,
                        'quantity': 1
                    }
                    self.target = self.model.inventory.pos
                    return

    def execute_mission(self):
        """Effectue une étape de la mission en cours."""
        if self.pos != self.target:
            self.move_toward(self.target)
            return

        if self.mission['type'] == 'collect' and self.cargo is None:
            donator = self.mission['source']
            food_name = self.mission['food']
            
            # Trouver l'objet FoodItem correspondant dans les dons
            matching_item = next((item for item in donator.donation_items if item.name == food_name), None)

            if matching_item:
                donator.donation_items.remove(matching_item)
                donator.donations[food_name] -= 1
                self.cargo = matching_item
                self.model.collected_food += 1
                self.target = self.mission['destination'].pos
            else:
                self.mission = None

        elif self.mission['type'] == 'deliver' and self.cargo:
            destination = self.mission['destination']

            if isinstance(destination, CentralInventory):
                success = destination.receive_item(self.cargo)
                if success:
                    self.model.delivered_food += 1
            elif isinstance(destination, Recipient):
                food_name = self.cargo.name
                if destination.demand_profile.get(food_name, 0) > 0:
                    destination.demand_profile[food_name] -= 1
                    self.model.inventory.current_load -= 1
                    self.model.delivered_food += 1

            self.cargo = None
            self.mission = None

    def move_toward(self, target):
        """Déplacement simple vers la cible (grille Manhattan)."""
        x, y = self.pos
        tx, ty = target
        if tx > x:
            x += 1
        elif tx < x:
            x -= 1
        elif ty > y:
            y += 1
        elif ty < y:
            y -= 1
        new_pos = (x, y)

        # Utilise la méthode de Mesa pour déplacer l'agent
        self.model.grid.move_agent(self, new_pos)
        self.pos = new_pos  # Mets à jour la position interne aussi (utile si d'autres objets l'utilisent)



class Donator(Agent):
    def __init__(self, unique_id, model, name, pos):
        super().__init__(unique_id, model)
        self.name = name
        self.pos = pos
        self.donations = {}  # dict[foodItem, int]

    def add_food_item(self, food_item, quantity: int):
        if not food_item.is_expired():
            if food_item in self.donations:
                self.donations[food_item] += quantity
            else:
                self.donations[food_item] = quantity


class InventoryAgent(Agent):
    def __init__(self, unique_id, model, capacity, current_load=0):
        super().__init__(unique_id, model)
        self.pos = (0, 0)  # Position fixe
        self.capacity = capacity
        self.current_load = current_load
        self.items = []

    def receive_item(self, food_item):
        if self.current_load < self.capacity:
            self.items.append(food_item)
            self.current_load += 1
            return True
        return False

class Recipient(Agent):
    def __init__(self, unique_id, model, name, pos, priority, distance):
        super().__init__(unique_id, model)
        self.name = name
        self.pos = pos
        self.demand_profile = {}  # dict[str, int]
        self.priority = priority
        self.distance = distance

    def __repr__(self):
        return f"Recipient({self.name})"

    def request_food(self):
        print(f"\n{self.name} is placing a food request.")
        while True:
            food_name = input("Enter food name (or type 'done' to finish): ").strip()
            if food_name.lower() == 'done':
                break
            try:
                if food_name in self.demand_profile:
                    print("You have already added this Food Item to the Demand list")
                    continue

                quantity = int(input(f"Enter quantity for {food_name}: ").strip())

                if quantity <= 0:
                    print("Quantity must be greater than 0.")
                    continue

                self.demand_profile[food_name] = quantity
                print(f"{food_name} request updated with quantity: {quantity}")
            except ValueError:
                print("Please enter a valid integer for quantity.")

        print("\nFood request completed.")

    def receive_food(self, food_items):
        for food_item in food_items:
            print(f"{self.name} has received {food_item.name}.")
            if food_item.name in self.demand_profile:
                del self.demand_profile[food_item.name]
                print(f"{food_item.name} removed from demand profile.")

    def view_demand(self):
        return self.demand_profile

    def get_recipient_priority(self):
        return self.priority
'''
# Example Testing Code:

# Create a Recipient object
recipient = Recipient(id=1, name="John Doe", location="123 Street")

# Recipient places a food request (interactive)
recipient.request_food()

# View the demand profile after the request
print(f"\nUpdated demand profile: {recipient.view_demand()}")

# Create example food items (with name, category, expiry date, nutritional value, and quantity)
food_item1 = FoodItem(name="Apple", category="Fruit", expiry_date=datetime(2025, 12, 15), nutritional_value=52.0, quantity=100,image_path='img.png',donation_time=datetime(2024, 12, 15),temperature=14.5, humidity=50, packaging_condition="emballage plastique",storage_duration=20.3)
food_item2 = FoodItem(name="Avocado", category="Fruit", expiry_date=datetime(2025, 12, 15), nutritional_value=52.0, quantity=100,image_path='img.png',donation_time=datetime(2024, 12, 15),temperature=14.5, humidity=50, packaging_condition="emballage plastique",storage_duration=20.3)

# The recipient receives the food items
recipient.receive_food([food_item1, food_item2])

# View the updated demand profile after receiving the food
print(f"\nUpdated demand profile after receiving food: {recipient.view_demand()}")
'''