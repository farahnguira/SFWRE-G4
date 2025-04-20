# Import necessary libraries
import os  # For interacting with the operating system and file paths
import csv  # For creating and writing to CSV files

# Define the root directory where images are stored (using raw string to avoid escape character issues)
image_root = r"C:\ISG\2BIS\probleme solving\DS2\data\images"
# Define where to save the output CSV file
csv_output_path = r"C:\ISG\2BIS\probleme solving\DS2\data\labels.csv"

# Debug message to show which path we're checking
print(f"Checking path: {image_root}")
# Verify the image directory exists
if not os.path.exists(image_root):
    # If path doesn't exist, show error and exit program
    print("Image path does not exist. Check your path.")
    exit()

# Open the CSV file for writing (creates new file or overwrites existing one)
with open(csv_output_path, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    # Write the header row to the CSV file
    writer.writerow(['filename', 'class'])  # Header with column names

    # Loop through each item in the root directory
    for item in os.listdir(image_root):
        # Create full path to the current item
        item_path = os.path.join(image_root, item)
        
        # Check if the current item is a directory (not a file)
        if os.path.isdir(item_path):
            # Handle regular class folders (not the 'fresh' folder)
            if item != 'fresh':
                # Loop through each image file in this class folder
                for img_file in os.listdir(item_path):
                    # Create relative path for the image (classname/filename)
                    relative_path = f"{item}/{img_file}"
                    # Write the image path and its class to CSV
                    writer.writerow([relative_path, item])
                    # Print confirmation message
                    print(f"{relative_path}, {item}")
            
            # Special handling for the 'fresh' folder
            else:
                # Store path to fresh folder
                fresh_path = item_path
                # Loop through subdirectories in fresh (fruit and vegetables)
                for sub_category in os.listdir(fresh_path):
                    # Create full path to the subcategory
                    sub_path = os.path.join(fresh_path, sub_category)
                    # Verify it's a directory (not a file)
                    if os.path.isdir(sub_path):
                        # Loop through each image in the subcategory
                        for img_file in os.listdir(sub_path):
                            # Create relative path (fresh/subcategory/filename)
                            relative_path = f"fresh/{sub_category}/{img_file}"
                            # Write to CSV with combined class name (fresh_subcategory)
                            writer.writerow([relative_path, f"fresh_{sub_category}"])
                            # Print confirmation message
                            print(f"{relative_path}, fresh_{sub_category}")

# Final confirmation message showing where CSV was saved
print(f"\nlabels.csv created at: {csv_output_path}")