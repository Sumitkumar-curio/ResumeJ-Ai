# Use a lightweight Python image as the base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app/backend

# Install system dependencies required by some Python packages (e.g., for pdfplumber, matplotlib)
# This is a common set; might need adjustment based on specific errors
RUN apt-get update && apt-get install -y     build-essential     libgl1-mesa-glx     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your FastAPI app runs on
EXPOSE $PORT

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]