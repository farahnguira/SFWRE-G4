from typing import List, Tuple, Dict
import copy
from datetime import datetime,timedelta
from food_item import FoodItem
from composants import *
from collections import defaultdict

def backtracking_food_distribution(
    available_food_items: List[FoodItem],
    requested_food_items: List[Tuple[str, int]],  # List of (food_name, requested_quantity)
    recipients: List[Recipient]
) -> Tuple[List[Tuple[FoodItem, Recipient, int]], float]:
    """
    Backtracking algorithm to find the best food distribution plan.
    
    Returns:
        - List of assignments (FoodItem, Recipient, quantity)
        - Total score of the assignment
    """
    best_assignment = []
    best_score = float('-inf')

    # Pre-process available food items
    valid_food_items = [f for f in available_food_items if f.expiry_date >= datetime.now()]
    sorted_food = sorted(valid_food_items, key=lambda f: f.expiry_date)

    # Create food inventory mapping
    food_name_map: Dict[str, List[FoodItem]] = {}
    for food in sorted_food:
        if food.name not in food_name_map:
            food_name_map[food.name] = []
        food_name_map[food.name].append(food)

    def backtrack(
        remaining_requests: List[Tuple[str, int]],
        current_assignment: List[Tuple[FoodItem, Recipient, int]],
        current_score: float,
        food_inventory: Dict[str, List[FoodItem]],
        recipient_demands: Dict[str, Dict[str, int]]  # recipient.name -> {food_name: remaining_demand}
    ):
        nonlocal best_assignment, best_score

        # Base case: all requests fulfilled
        if not remaining_requests:
            if current_score > best_score:
                best_score = current_score
                best_assignment = copy.deepcopy(current_assignment)
            return
            
        current_food_name, current_request_qty = remaining_requests[0]
        
        # Skip if no available items for this food
        if current_food_name not in food_inventory or not food_inventory[current_food_name]:
            return backtrack(
                remaining_requests[1:],
                current_assignment,
                current_score,
                food_inventory,
                recipient_demands
            )

        # Try to assign to each recipient
        for recipient in recipients:
            # Check recipient's remaining demand for this food
            recipient_remaining = recipient_demands[recipient.name].get(current_food_name, 0)
            if recipient_remaining <= 0:
                continue

            # Determine possible quantity to assign
            available_qty = sum(f.quantity for f in food_inventory[current_food_name])
            assign_qty = min(current_request_qty, recipient_remaining, available_qty)

            if assign_qty <= 0:
                continue

            # Find best food items to use (prioritize soon-to-expire first)
            remaining_to_assign = assign_qty
            items_used = []
            
            for food_item in food_inventory[current_food_name]:
                if remaining_to_assign <= 0:
                    break
                
                use_qty = min(food_item.quantity, remaining_to_assign)
                items_used.append((food_item, use_qty))
                remaining_to_assign -= use_qty

            # Calculate score for this assignment
            assignment_score = 0
            for food_item, qty in items_used:
                food_score = food_item.get_priority_score(donation_size=qty, distance=recipient.distance)
                recipient_score = recipient.get_recipient_priority()
                assignment_score += food_score * recipient_score / (1 + recipient.distance)

            # Make the assignment
            new_assignment = current_assignment.copy()
            for food_item, qty in items_used:
                new_assignment.append((food_item, recipient, qty))

            # Update inventory
            new_inventory = copy.deepcopy(food_inventory)
            for food_item, qty in items_used:
                for i, f in enumerate(new_inventory[current_food_name]):
                    if f == food_item:
                        f.quantity -= qty
                        if f.quantity == 0:
                            new_inventory[current_food_name].pop(i)
                        break

            # Update recipient demands
            new_demands = copy.deepcopy(recipient_demands)
            new_demands[recipient.name][current_food_name] -= assign_qty

            # Proceed to next request
            backtrack(
                remaining_requests[1:],
                new_assignment,
                current_score + assignment_score,
                new_inventory,
                new_demands
            )

    # Initialize recipient demands
    initial_demands = {}
    for recipient in recipients:
        initial_demands[recipient.name] = recipient.view_demand().copy()

    # Start backtracking with all requests
    backtrack(
        remaining_requests=requested_food_items,
        current_assignment=[],
        current_score=0.0,
        food_inventory=food_name_map,
        recipient_demands=initial_demands
    )

    return best_assignment, best_score


# Dummy Recipient class
class DummyRecipient(Recipient):
    def __init__(self, id, name, location, priority, demand,distance):
    
        super().__init__(id, name, location, demand,distance)
        self.priority = priority
        self.demand_profile = demand
        self.distance=distance

    def get_recipient_priority(self):
        return self.priority




def test_simple_backtracking():
    """Test function for the backtracking algorithm."""
    now = datetime.now()
    
    # Create test food items
    food_items = [
        FoodItem("Rice", "Grain", now + timedelta(days=10), 0.8, 10, "", now, 20, 30, "Good", 1),
        FoodItem("Milk", "Dairy", now + timedelta(days=1), 0.9, 5, "", now, 5, 50, "Good", 2),
        FoodItem("Apple", "Fruit", now + timedelta(days=3), 0.7, 8, "", now, 10, 60, "Average", 1.5),
        FoodItem("Chicken", "Meat", now - timedelta(days=1), 1.0, 4, "", now, 2, 40, "Poor", 3),  # expired
        FoodItem("Beans", "Legume", now + timedelta(days=15), 0.6, 6, "", now, 18, 35, "Good", 1)

    ]
     # Create test recipients
    recipients=[
        DummyRecipient(
        id=1, name="Shelter A", location="Loc A", 
        priority=5, demand={"Rice": 5, "Milk": 2, "Apple": 2},distance=18
        ),
        DummyRecipient(
            id=2, name="Family B", location="Loc B", 
            priority=3, demand={"Apple": 3, "Beans": 4},distance=25
        )]
    
    def get_all_demands(recipients: list) -> List[Tuple[str, int]]:
        all_demands = []

        for recipient in recipients:
            demands = recipient.view_demand()
            print("demands 1111111 :", demands)
            
            for food_name, quantity in demands.items():
                all_demands.append((food_name, quantity))

        return all_demands
                
    demands=get_all_demands(recipients)
   # Run the algorithm
    final_plan, final_score = backtracking_food_distribution(
        available_food_items=food_items,
        requested_food_items=demands,
        recipients=recipients   
    )
    print("demands :")
    print(demands)

    # Print results
    print("\n=== FINAL DISTRIBUTION PLAN ===")
    print(final_plan)
    
    print(f"Total Score: {final_score:.2f}")
    
    # Verify results
   #print("\n=== TEST VERIFICATION ===")
   # verify_distribution(final_plan, food_items, [recipient1, recipient2])


'''
def verify_distribution(plan, items, recipients=):
    """Helper function to verify distribution results."""
    # Check expired items weren't distributed
    expired_items = {item.name for item in items if item.expiry_date < datetime.now()}
    expired_distributed = any(
        item.name in expired_items
        for allocations in plan.values()
        for item in allocations
    )
    print(f"Expired items check: {'PASS' if not expired_distributed else 'FAIL'}")
    
    # Check quantities don't exceed available stock
    available = {item.name: item.quantity for item in items}
    distributed = {}
    for allocations in plan.values():
        for item, qty in allocations.items():
            distributed[item.name] = distributed.get(item.name, 0) + qty
    
    quantity_checks = all(
        distributed.get(name, 0) <= available.get(name, 0)
        for name in distributed
    )
    print(f"Quantity checks: {'PASS' if quantity_checks else 'FAIL'}")
    
    # Check demands were respected
    recipient_demands = {
        r.id: r.view_demand for r in recipients
    }
    demand_checks = all(
        sum(
            qty for rid, allocations in plan.items()
            for item, qty in allocations.items()
            if item.name == food and rid in recipient_demands
        ) <= sum(
            r.view_demand.get(food, 0)
            for r in recipients
        )
        for food in distributed
    )
    print(f"Demand checks: {'PASS' if demand_checks else 'FAIL'}")
    '''

if __name__ == "__main__":
    test_simple_backtracking()