import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_data(input_path="amine/data/raw_data.csv", output_path="amine/data/cleaned_data.csv"):
    """Clean the raw data and save to cleaned_data.csv."""
    try:
        # Load the raw data
        df = pd.read_csv(input_path, parse_dates=["donation_date", "expiry_date"])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.dropna(subset=["donation_date", "expiry_date", "type", "quantity", "priority"])
        
        # Ensure data types
        df["quantity"] = df["quantity"].astype(int)
        df["priority"] = df["priority"].astype(int)
        df["type"] = df["type"].astype(str)
        
        # Remove invalid quantities
        df = df[df["quantity"] >= 0]
        
        # Ensure donation_date is before expiry_date
        df = df[df["donation_date"] <= df["expiry_date"]]
        
        # Sort by expiry_date
        df = df.sort_values("expiry_date")
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to clean data: {e}")
        raise

if __name__ == "__main__":
    clean_data()