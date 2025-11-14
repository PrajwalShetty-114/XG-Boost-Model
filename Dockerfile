# Use an official Python runtime as a parent image
FROM python:3.11-slim

# 1. Install system dependencies, including git and git-lfs
# We run this first to leverage build caching
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install Git LFS
RUN git lfs install

# 4. Copy and install Python requirements
# This is done *before* copying the rest of the app for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project (including .git) into the container
COPY . .

# 6. Pull the LFS files
# This will download the .pkl file
RUN git lfs pull

# 7. Set the command to run the app using gunicorn
# Render will automatically set the PORT environment variable
# We bind to 0.0.0.0 so it's accessible from outside the container
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT"]