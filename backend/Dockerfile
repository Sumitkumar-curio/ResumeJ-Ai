# Use a lightweight Python image as the base
FROM python:3.9-slim-bullseye

# Set the working directory in the container
WORKDIR /app/backend

# Install system dependencies required by some Python packages (e.g., for pdfplumber, matplotlib)
# This is a common set; might need adjustment based on specific errors
RUN apt-get update && apt-get install -y     build-essential     libgl1-mesa-glx     libgirepository1.0-dev     libcairo2-dev     pkg-config     python3-dev     poppler-utils     tesseract-ocr     libxml2-dev     libxslt-dev     libgomp1     libatlas-base-dev     ffmpeg libsm6 libxext6     rustc cargo     swig     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt -v

# Copy the rest of your application code
COPY . .

# Expose the port your FastAPI app runs on
EXPOSE $PORT

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]