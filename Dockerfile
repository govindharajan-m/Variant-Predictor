# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user (Recommended by Hugging Face Spaces and for general security)
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Set the working directory to the user's home app directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app
COPY --chown=user . $HOME/app

# Generate the ML model during the build phase
# (This ensures the ml_model/ directory and files are available for inference)
RUN python train_model.py

# Expose port 7860 (Hugging Face default, also works for general Docker)
EXPOSE 7860

# Command to run the Gunicorn server
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:7860", "--workers", "2", "--threads", "4", "--timeout", "120"]
