# ResumeJ AI Backend

This directory contains the Python backend for the ResumeJ AI application.

## Setup

1.  **Install Python:** Make sure you have Python 3.8 or higher installed.
2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the server

Once the setup is complete, you can start the FastAPI server:

```bash
uvicorn main:app --reload
```

The server will be running at `http://localhost:8000`.
