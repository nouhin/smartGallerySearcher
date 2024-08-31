# Use an official Python runtime as the parent image
FROM python:3.14-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install the dependencies first so code changes don't invalidate this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container at /app
COPY . .

# Run agent.py when the container launches, arguments are passed to it
ENTRYPOINT ["python", "agent.py"]
