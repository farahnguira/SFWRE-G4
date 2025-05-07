from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import numpy as np
from typing import List, Dict
from assign_items import find_optimal_distribution 
import math
from backtracking_recipients import backtracking_food_distribution
from composants import *

class FoodDistributionModel(Model):
   
    
    def __init__(self, donators: List[Donator], recipients: List[Recipient], inventory: InventoryAgent
):
        super().__init__()
        # Configuration de la grille
        self.grid = MultiGrid(50, 50, torus=False)
        self.schedule = RandomActivation(self)
        
        # Stockage des objets métiers
        self.donators = donators
        self.recipients = recipients
        self.inventory = inventory
        
        # Placement des entités
        self.place_entities()
        
        # Création des transporteurs
        self.create_transport_agents(3)
        
        # Statistiques
        self.collected_food = 0
        self.delivered_food = 0
    
    def place_entities(self):
        """Place les entités sur la grille."""
        # Placement des donateurs (aléatoire)
        for donator in self.donators:
            x = self.random.randrange(10, 40)
            y = self.random.randrange(10, 40)
            donator.pos = (x, y)  # Mise à jour de la position
            self.grid.place_agent(DonatorAgent(self.next_id(), self, donator), (x, y))
        
        # Placement de l'inventaire (centre)
        self.inventory.pos = (25, 25)
        self.grid.place_agent(InventoryAgent(self.next_id(), self, self.inventory), (25, 25))
        
        # Placement des bénéficiaires (périphérie)
        for i, recipient in enumerate(self.recipients):
            angle = 2 * np.pi * i / len(self.recipients)
            x = 25 + int(20 * np.cos(angle))
            y = 25 + int(20 * np.sin(angle))
            recipient.location = (x, y)
            self.grid.place_agent(Recipient(self.next_id(), self, recipient), (x, y))
    
    def create_transport_agents(self, n: int):
        """Crée n agents de transport."""
        for _ in range(n):
            ta = TransportAgent(self.next_id(), self)
            x, y = self.random.randrange(50), self.random.randrange(50)
            self.grid.place_agent(ta, (x, y))
            self.schedule.add(ta)
    
    def step(self):
        
        if self.schedule.time % 10 == 0:  # Exécute tous les 10 pas
            self.run_donation_solver()
        if self.schedule.time % 5 == 0:  # Exécute tous les 10 pas
            self.run_solver()
        self.schedule.step()

    def run_solver(self):
        """Génère les plans de distribution avec backtracking et assigne aux transporteurs."""
        food_items = self.inventory.items  # Liste de FoodItem
        requests = []

        for recipient in self.recipients:
            for food_name, quantity in recipient.demand_profile.items():
                if quantity > 0:
                    requests.append((food_name, quantity))

        assignments, score = backtracking_food_distribution(
            available_food_items=food_items,
            requested_food_items=requests,
            recipients=self.recipients
        )

        self.pending_deliveries = []  # Liste de (FoodItem, Recipient, quantity)
        self.distribution_log = []    # Liste de dictionnaires pour le log

        for food_item, recipient, qty in assignments:
            self.pending_deliveries.append((food_item, recipient, qty))

            # Log sous forme de dictionnaire
            self.distribution_log.append({
                'food_item': food_item.name,
                'recipient': recipient.name,
                'recipient_id': recipient.id,
                'quantity': qty,
                'expiration_date': food_item.expiry_date
            })

        inventory_pos = self.inventory.pos
        transport_agents = [agent for agent in self.schedule.agents if isinstance(agent, TransportAgent)]

        for delivery in self.pending_deliveries:
            food_item, recipient, qty = delivery
            best_agent = min(transport_agents, key=lambda a: math.dist(a.pos, inventory_pos))
            best_agent.add_delivery_mission(food_item, recipient, qty)

    def run_donation_solver(self):
        """Utilise find_optimal_distribution() pour créer et assigner les missions de collecte."""

        best_assignments, score = find_optimal_distribution(
            donators=self.donators,
            inventory=self.inventory
        )

        self.donation_log = []
        transport_agents = [agent for agent in self.schedule.agents if isinstance(agent, TransportAgent)]

        for food_item, donator, qty in best_assignments:
            for _ in range(qty):  # Une mission par unité
                best_agent = min(transport_agents, key=lambda a: math.dist(a.pos, donator.pos))
                best_agent.add_collection_mission(food_item, donator, self.inventory)

                self.donation_log.append({
                    "food": food_item.name,
                    "from": donator.name,
                    "to": "inventory",
                    "agent_id": best_agent.unique_id
                })


    # Fonction de simulation
    def run_simulation(donators: List[Donator], 
                    recipients: List[Recipient], 
                    inventory: CentralInventory,
                    steps: int = 100) -> Dict:
        """Lance la simulation complète."""
        model = FoodDistributionModel(donators, recipients, inventory)
        for _ in range(steps):
            model.step()
        
        return {
            'collected': model.collected_food,
            'delivered': model.delivered_food,
            'remaining_donations': sum(sum(d.donations.values()) for d in model.donators),
            'unsatisfied_demand': sum(sum(r.demand_profile.values()) for r in model.recipients)
        }
