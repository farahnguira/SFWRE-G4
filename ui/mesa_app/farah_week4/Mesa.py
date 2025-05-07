from composants import *
from food_item import FoodItem
from MesaApp import run_simulation
from datetime import datetime 


# Exemple de données
food_riz = FoodItem(
    name="riz", category="féculent", expiry_date=datetime(2025, 6, 1),
    nutritional_value=350, quantity=5, image_path="riz.png",
    donation_time=datetime.now(), temperature=22.0, humidity=45.0,
    packaging_condition="bon", storage_duration=7.0
)

food_pates = FoodItem(
    name="pâtes", category="féculent", expiry_date=datetime(2025, 6, 5),
    nutritional_value=300, quantity=3, image_path="pates.png",
    donation_time=datetime.now(), temperature=22.0, humidity=50.0,
    packaging_condition="bon", storage_duration=7.0
)

food_pain = FoodItem(
    name="pain", category="boulangerie", expiry_date=datetime(2025, 5, 10),
    nutritional_value=250, quantity=4, image_path="pain.png",
    donation_time=datetime.now(), temperature=20.0, humidity=40.0,
    packaging_condition="moyen", storage_duration=2.0
)

# Définition des donateurs avec un dictionnaire {FoodItem: quantité}
donators = [
    Donator("Supermarché A", {food_riz: 5, food_pates: 3}),
    Donator("Boulangerie B", {food_pain: 4}),
]


recipient1 = Recipient(
id=1,
name="Famille X",
location="Rue A",
priority=0.9,
distance=3.5
)
recipient1.demand_profile={"riz": 2, "pain": 1}
recipient2 = Recipient(
id=2,
name="Famille Y",
location="Rue B",
priority=0.7,
distance=5.0
)
recipient2.demand_profile={"pâtes": 2}
recipients = [recipient1 , recipient2 ]

inventory = CentralInventory(capacity=10, current_load=5)

# Lancer la simulation
resultats = run_simulation(donators, recipients, inventory, steps=50)

# Afficher les résultats
print("Résultats de la simulation :")
print(resultats)
