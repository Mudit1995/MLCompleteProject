FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Update the package list and install AWS CLI
RUN apt update -y && apt install awscli -y

# Install Python dependencies
RUN pip install --no-cache-dir -r requirments.txt

# Command to run the application
CMD ["python3", "app.py"]
