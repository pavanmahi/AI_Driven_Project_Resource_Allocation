# AI-Driven Project Resource Mapping (Backend)

This project is an AI-driven resource mapping system designed to assist resource allocation managers in efficiently assigning employees to projects. It focuses on optimizing resource utilization and project success by intelligently matching employee skills and availability with project requirements.

## Project Overview

The backend of this system leverages AI to facilitate intelligent resource allocation. It includes components for:

-   **Employee Data Management**: Storing and managing employee profiles, skills, and availability.
-   **Project Data Management**: Handling project details, requirements, and timelines.
-   **LLM Service**: Utilizing large language models for advanced processing and analysis.
-   **Vector Database**: Storing and retrieving vectorized data for efficient similarity searches and matching.
-   **Matching Engine**: The core AI component responsible for recommending optimal employee-to-project mappings.
-   **API**: Providing endpoints for frontend interaction and data exchange.

## Technologies Used

-   **Python**: Primary programming language.
-   **FastAPI**: For building the backend API.
-   **Chroma DB**: For vector database functionalities.
-   **Hugging Face Transformers**: (Potentially) for LLM integration.

## Project Structure

-   <mcfolder name="backend" path="d:\start web\Hackathon\Gen_Ai\backend"></mcfolder>: Contains all backend services and logic.
    -   <mcfolder name="app" path="d:\start web\Hackathon\Gen_Ai\backend\app"></mcfolder>: Main application entry point and API definitions.
        -   <mcfile name="main.py" path="d:\start web\Hackathon\Gen_Ai\backend\app\main.py"></mcfile>: FastAPI application setup.
        -   <mcfolder name="api" path="d:\start web\Hackathon\Gen_Ai\backend\app\api"></mcfolder>: API routes and endpoints.
        -   <mcfolder name="services" path="d:\start web\Hackathon\Gen_Ai\backend\app\services"></mcfolder>: Business logic and service implementations.
        -   <mcfile name="websocket_manager.py" path="d:\start web\Hackathon\Gen_Ai\backend\app\websocket_manager.py"></mcfile>: Manages WebSocket connections.
    -   <mcfolder name="chroma_db" path="d:\start web\Hackathon\Gen_Ai\backend\chroma_db"></mcfolder>: ChromaDB data storage.
    -   <mcfolder name="data" path="d:\start web\Hackathon\Gen_Ai\backend\data"></mcfolder>: Data generation scripts and datasets.
        -   <mcfile name="Employee.csv" path="d:\start web\Hackathon\Gen_Ai\backend\data\Employee.csv"></mcfile>, <mcfile name="project_description_for_er.csv" path="d:\start web\Hackathon\Gen_Ai\backend\data\project_description_for_er.csv"></mcfile>: Sample datasets.
    -   <mcfile name="llm_service.py" path="d:\start web\Hackathon\Gen_Ai\backend\llm_service.py"></mcfile>: Integrates and manages LLM interactions.
    -   <mcfile name="matching_engine.py" path="d:\start web\Hackathon\Gen_Ai\backend\matching_engine.py"></mcfile>: Core logic for matching employees to projects.
    -   <mcfile name="models.py" path="d:\start web\Hackathon\Gen_Ai\backend\models.py"></mcfile>: Data models for employees and projects.
    -   <mcfile name="vector_db.py" path="d:\start web\Hackathon\Gen_Ai\backend\vector_db.py"></mcfile>: Handles interactions with the vector database.
    -   <mcfile name="requirements.txt" path="d:\start web\Hackathon\Gen_Ai\backend\requirements.txt"></mcfile>: Python dependencies.
-   <mcfolder name="Tests" path="d:\start web\Hackathon\Gen_Ai\Tests"></mcfolder>: Contains test cases for the API and other components.

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Gen_Ai
    ```
2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```

## Running the Application

To start the backend server, navigate to the `backend` directory and run:

```bash
cd backend
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000` (or another specified port).

## Usage

Detailed API documentation will be available at `http://localhost:8000/docs` once the server is running. This will provide information on available endpoints, request/response formats, and how to interact with the system.