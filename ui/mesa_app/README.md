# Smart Food Waste and Redistribution Ecosystem - Mesa App 
 
This directory contains the Mesa-based simulation for the Food Redistribution Model. 
 
## Setup Instructions 
1. Navigate to the directory: 
   ``` 
   cd path/to/SFWRE-G4/ui/mesa_app 
   ``` 
2. Create and activate the virtual environment: 
   ``` 
   python -m venv .venv 
   .venv\Scripts\activate 
   ``` 
3. Install dependencies: 
   ``` 
   pip install -r requirements.txt 
   ``` 
4. Run the simulation: 
   ``` 
   python server.py 
   ``` 
   Open a browser and go to `http://127.0.0.1:8521` to view the visualization. 
 
## Project Structure 
- `.venv/`: Virtual environment (ignored by Git) 
- `model.py`: Defines the `FoodModel` and agent portrayal 
- `server.py`: Launches the Mesa visualization server 
- `requirements.txt`: Lists dependencies 
- `.gitignore`: Ignores `.venv` and other generated files 
