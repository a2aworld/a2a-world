FROM python:3.12-slim

# Install curl, which is required for the health check to run
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the server runs on
EXPOSE 8080

# Command to run the Python server
CMD ["python", "server.py"]
