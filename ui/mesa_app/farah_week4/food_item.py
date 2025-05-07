from datetime import datetime

class FoodItem:
    def __init__(self, name:str, category:str, expiry_date:datetime, nutritional_value:float, quantity:int, 
                 image_path:str, donation_time:datetime, temperature:float, humidity:float, packaging_condition: str, 
                 storage_duration:float):
        
        self.name = name
        self.category = category
        self.expiry_date = expiry_date
        self.nutritional_value = nutritional_value
        self.quantity = quantity
        self.image_path = image_path
        self.donation_time = donation_time
        self.temperature = temperature
        self.humidity = humidity
        self.packaging_condition = packaging_condition
        self.storage_duration = storage_duration


    def is_expired(self) -> bool:
        """Check if the food item has expired."""
        return datetime.now() > self.expiry_date   
     
    def __repr__(self):
        return f"FoodItem({self.name})"
    

    def is_perishable(self)->bool:
        """
        Determines if the food item is perishable based on its category.
        """
        perishable_categories = ['Fruit', 'Vegetable', 'Dairy', 'Meat', 'Fish'] #has to be updated based on the dataset
        return self.category in perishable_categories

    def get_priority_score(self, donation_size: int, distance: float) -> float:
        """
        Calculate the priority score for the food item based on its attributes.
        
        :param donation_size: The expected size of the donation (could be in kg, pieces, etc.)
        :param distance: The distance between the donation location and the recipient (in km)
        :return: A priority score for the food item.
        """
        
        # Calculate the number of days left until expiry
        days_left = (self.expiry_date - datetime.now()).days
        
        # Expiry factor (closer to expiry = higher priority)
        expiry_factor = 1 / (days_left + 1)  # The closer to expiry, the higher the priority
        
        # Nutritional value factor (higher nutritional value = higher priority)
        nutritional_factor = self.nutritional_value

        '''
        # Quantity factor (larger quantity = higher priority)
        quantity_factor = self.quantity / 100  # Larger donations are prioritized
        '''

        # Storage duration factor (older food = higher priority)
        storage_factor = self.storage_duration 
        
        # Perishability factor (perishable items = higher priority)
        perishability_factor = 2 if self.is_perishable() else 1
        
        # Donation size factor (larger donations have higher priority)
        donation_size_factor = donation_size / self.quantity  # Larger donations are given higher priority
        
        
        # Final priority score calculation (adjust weights as needed)
        priority_score = (
            (nutritional_factor * 0.3) +
            (expiry_factor * 0.3) +
            (storage_factor * 0.1) +
            (perishability_factor * 0.2) +
            (donation_size_factor * 0.1) 
        )
        
        return priority_score
