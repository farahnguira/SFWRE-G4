# Smart Food Waste and Redistribution Ecosystem 
 
## Project Overview 
This project aims to reduce food waste by simulating redistribution using agent-based modeling (Mesa) and data visualization (Dash). The Mesa app models food redistribution, while the Dash app will provide a dashboard for insights. 
 
## Setup (venv & Docker) 
### Virtual Environment 
1. Navigate to `ui/mesa_app`: 
   ``` 
   cd ui/mesa_app 
   ``` 
2. Create and activate a virtual environment: 
   ``` 
   python -m venv .venv 
   .venv\Scripts\activate 
   ``` 
3. Install dependencies: 
   ``` 
   pip install -r requirements.txt 
   ``` 
### Docker (Optional) 
- Instructions for Docker setup will be added soon. 
 
## UML Diagram Links 
- Sequence Diagram: [Link to be added] 
- Use Case Diagram: [Link to be added] 
- Class Diagram: [Link to be added] 
*Note*: UML diagrams are available in PlantUML format. Contact the team for .puml files to render in tools like StarUML or Visual Paradigm. 
 
## How to Run UI Apps 
### Mesa App 
1. Navigate to `ui/mesa_app`: 
   ``` 
   cd ui/mesa_app 
   ``` 
2. Activate the virtual environment: 
   ``` 
   .venv\Scripts\activate 
   ``` 
3. Run the server: 
   ``` 
   python server.py 
   ``` 
4. Open `http://127.0.0.1:8521` in a browser. 
### Dash App 
1. Navigate to `ui/dash_app`: 
   ``` 
   cd ui/dash_app 
   ``` 
2. Activate the virtual environment from `mesa_app`: 
   ``` 
   ..\mesa_app\.venv\Scripts\activate 
   ``` 
3. Run the app: 
   ``` 
   python app.py 
   ``` 
4. Open `http://127.0.0.1:8050` in a browser. 
 
## Contact & Contribution Guide 
- **Contact**: Reach out to the team via GitHub issues or email (to be added). 
- **Contribution**: Fork the repository, create a branch, and submit a pull request. Ensure to follow the coding style and test your changes. 
