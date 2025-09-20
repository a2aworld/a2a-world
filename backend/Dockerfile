FROM python:3.12-slim

# Install system dependencies including curl for health checks
# and build-essentials for python packages that need compilation.
# GDAL is required for geospatial libraries.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    build-essential \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# THIS IS THE FIX: Set the PYTHONPATH environment variable.
# This tells Python to look for modules inside the /app directory.
ENV PYTHONPATH=/app

# Copy requirements and install Python dependencies
COPY requirements.txt .
# We also need geo-dependencies for the seeding script
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pandas psycopg2-binary shapely geoalchemy2

# Copy the entire project context to make all modules available
COPY . .

# Expose the port the server runs on
EXPOSE 8000

# Command to run the Uvicorn server for your FastAPI app.
# This command will now work because of the PYTHONPATH we set.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]