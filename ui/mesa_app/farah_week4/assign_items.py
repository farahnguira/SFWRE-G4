from typing import List, Dict, Tuple, Optional
from datetime import datetime , timedelta
import copy
import math
from food_item import FoodItem
from composants import *

def find_optimal_distribution(
    donators: List['Donator'],
    inventory: 'CentralInventory'
) -> Tuple[List[Tuple['FoodItem', 'Donator', int]], float]:
    """
    Version finale avec :
    - donations en dict[FoodItem, int]
    - distance calculée avec abs(inventory.pos - donator.pos)
    """

    best_assignment = []
    best_score = float('-inf')

    # 1. Structurer les donations disponibles
    donations_pool: List[Tuple['FoodItem', 'Donator', int]] = []
    for donator in donators:
        for food_item, quantity in donator.donations.items():
            if quantity > 0:
                donations_pool.append((food_item, donator, quantity))

    # 2. Trier par distance croissante 
    donations_pool.sort(
        key=lambda x: abs(inventory.pos - x[1].pos)
    )

    def backtrack(
        index: int,
        current_assignment: List[Tuple['FoodItem', 'Donator', int]],
        current_score: float,
        remaining_capacity: float
    ):
        nonlocal best_assignment, best_score

        if index >= len(donations_pool) or remaining_capacity <= 0:
            if current_score > best_score:
                best_score = current_score
                best_assignment = copy.deepcopy(current_assignment)
            return

        food_item, donator, available_qty = donations_pool[index]

        distance = abs(inventory.pos - donator.pos)
        unit_score = 1 / (distance + 1)

        take = min(available_qty, remaining_capacity)
        if take > 0:
            backtrack(
                index + 1,
                current_assignment + [(food_item, donator, take)],
                current_score + (take * unit_score),
                remaining_capacity - take
            )

        # Ne rien prendre
        backtrack(
            index + 1,
            current_assignment,
            current_score,
            remaining_capacity
        )

    # Lancement
    backtrack(0, [], 0, inventory.capacity)

    return best_assignment, best_score
'''
import unittest
from datetime import datetime, timedelta
from typing import List, Tuple

class TestFoodDistribution(unittest.TestCase):
    def setUp(self):
        # Create test donators
        self.donator1 = Donator("Farm A", 40.7128)
        self.donator2 = Donator("Farm H", 10.7128)
        
        # Create test food items
        today = datetime.now()
        self.apples1 = FoodItem(
            name="Apples",
            category="Fruit",
            expiry_date=today + timedelta(days=5),
            nutritional_value=0.8,
            quantity=50,
            image_path="apples1.jpg",
            donation_time=today,
            temperature=4.0,
            humidity=0.6,
            packaging_condition="good",
            storage_duration=2
        )
        
        self.apples2 = FoodItem(
            name="Apples",
            category="Fruit",
            expiry_date=today + timedelta(days=3),  # More urgent
            nutritional_value=0.9,
            quantity=30,
            image_path="apples2.jpg",
            donation_time=today,
            temperature=3.5,
            humidity=0.5,
            packaging_condition="excellent",
            storage_duration=1
        )
        
        self.milk = FoodItem(
            name="Milk",
            category="Dairy",
            expiry_date=today + timedelta(days=2),  # Most urgent
            nutritional_value=0.95,
            quantity=20,
            image_path="milk.jpg",
            donation_time=today,
            temperature=2.0,
            humidity=0.4,
            packaging_condition="good",
            storage_duration=1
        )
        
        # Add donations to donators
        self.donator1.donations = [self.apples1, self.milk]
        self.donator2.donations = [self.apples2]
        
        # Create central inventory
        self.inventory = CentralInventory(40.7135, 100)  # Capacity of 100 units

    def test_basic_distribution(self):
        """Test basic distribution scenario"""
        requested_items = [("Apples", 70), ("Milk", 10)]
        
        assignments, score = find_optimal_distribution(
            donators=[self.donator1, self.donator2],
            inventory=self.inventory,
            requested_items=requested_items
        )
        
        # Verify assignments
        self.assertGreater(len(assignments), 0)
        self.assertGreater(score, 0)
        
        # Check if all apples were taken from donator2 first (more urgent)
        apples_from_donator2 = [a for a in assignments 
                              if a[1].name == "Farm B" and a[0].name == "Apples"]
        self.assertGreater(len(apples_from_donator2), 0)
        
        # Check if milk was assigned
        milk_assignments = [a for a in assignments if a[0].name == "Milk"]
        self.assertEqual(len(milk_assignments), 1)
        self.assertEqual(milk_assignments[0][2], 10)  # Full milk request fulfilled

    def test_capacity_constraint(self):
        """Test that inventory capacity is respected"""
        # Request more than capacity
        requested_items = [("Apples", 200)]
        
        assignments, score = find_optimal_distribution(
            donators=[self.donator1, self.donator2],
            inventory=self.inventory,
            requested_items=requested_items
        )
        
        # Calculate total assigned quantity
        total_assigned = sum(a[2] for a in assignments)
        self.assertLessEqual(total_assigned, self.inventory.capacity)

    def test_perishable_priority(self):
        """Test that perishable items get higher priority"""
        requested_items = [("Apples", 30), ("Milk", 20)]
        
        assignments, score = find_optimal_distribution(
            donators=[self.donator1, self.donator2],
            inventory=self.inventory,
            requested_items=requested_items
        )
        
        # Milk should be assigned first (more perishable)
        first_assignment = assignments[0]
        self.assertEqual(first_assignment[0].name, "Milk")

    def test_expired_items(self):
        """Test that expired items are not distributed"""
        # Create expired item
        expired_item = FoodItem(
            name="Expired Bread",
            category="Bakery",
            expiry_date=datetime.now() - timedelta(days=1),
            nutritional_value=0.7,
            quantity=10,
            image_path="bread.jpg",
            donation_time=datetime.now() - timedelta(days=2),
            temperature=3.0,
            humidity=0.5,
            packaging_condition="fair",
            storage_duration=1
        )
        self.donator1.donations.append(expired_item)
        
        requested_items = [("Expired Bread", 5)]
        
        assignments, score = find_optimal_distribution(
            donators=[self.donator1],
            inventory=self.inventory,
            requested_items=requested_items
        )
        
        self.assertEqual(len(assignments), 0)  # No assignments for expired items

    def test_partial_fulfillment(self):
        """Test when requests can only be partially fulfilled"""
        # Request more apples than available
        requested_items = [("Apples", 100)]
        
        assignments, score = find_optimal_distribution(
            donators=[self.donator1, self.donator2],
            inventory=self.inventory,
            requested_items=requested_items
        )
        
        # Should assign all available apples (50 + 30 = 80)
        total_apples = sum(a[2] for a in assignments if a[0].name == "Apples")
        self.assertEqual(total_apples, 80)

if __name__ == "__main__":
    unittest.main()
    
'''

# Test
# Create test donators
donator1 = Donator("Farm A", 40.7128)
donator2 = Donator("Farm H", 10.7128)
    
# Create test food items
today = datetime.now()
apples1 = FoodItem(
    name="Apples",
    category="Fruit",
    expiry_date=today + timedelta(days=5),
    nutritional_value=0.8,
    quantity=50,
    image_path="apples1.jpg",
    donation_time=today,
    temperature=4.0,
    humidity=0.6,
    packaging_condition="good",
    storage_duration=2
)

apples2 = FoodItem(
    name="Apples",
    category="Fruit",
    expiry_date=today + timedelta(days=3),  # More urgent
    nutritional_value=0.9,
    quantity=30,
    image_path="apples2.jpg",
    donation_time=today,
    temperature=3.5,
    humidity=0.5,
    packaging_condition="excellent",
    storage_duration=1
)

milk = FoodItem(
    name="Milk",
    category="Dairy",
    expiry_date=today + timedelta(days=2),  # Most urgent
    nutritional_value=0.95,
    quantity=20,
    image_path="milk.jpg",
    donation_time=today,
    temperature=2.0,
    humidity=0.4,
    packaging_condition="good",
    storage_duration=1
)

# Add donations to donators
donator1.donations = {"Apples": apples1.quantity, "Milk": milk.quantity}  # Dict[str,int]
donator2.donations = {"Apples": apples2.quantity}   

# Create central inventory
inventory = CentralInventory(40.7135, 100)  # Capacity of 100 units


assignments, score = find_optimal_distribution([donator1, donator2], inventory)
print("Assignments:")
for item, donor, qty in assignments:
    print(f"- {qty} {item} from {donor.name}")
print(f"Total score: {score}")