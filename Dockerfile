# Use an official Python runtime as a parent image
FROM python:3.11-slim

# 1. Install system dependencies, including git and git-lfs
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install Git LFS
RUN git lfs install --skip-smudge

# 4. Copy ONLY the Git and LFS tracking files first
# This gives the container the "instructions" for LFS
COPY .git ./.git
COPY .gitattributes .

# 5. Now, pull the LFS files (the .pkl, .keras, etc.)
# This command will now work because the .git folder is present
RUN git lfs pull

# 6. Copy the rest of your application code
# (main.py, requirements.txt, data/ folder, etc.)
COPY . .

# 7. Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# 8. Set the command to run the app using gunicorn
# Render will automatically set the PORT environment variable
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT"]
