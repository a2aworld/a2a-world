#!/bin/bash

# Terra Constellata Backend Startup Script

echo "Starting Terra Constellata backend services..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Using default values."
fi

# Start services
docker-compose up -d

echo "Services starting up..."
echo "PostgreSQL (PostGIS): http://localhost:${POSTGRES_PORT:-5432}"
echo "ArangoDB: http://localhost:${ARANGO_PORT:-8529}"
echo "A2A Protocol Server: http://localhost:${A2A_PORT:-8080}"
echo "Backend API: http://localhost:${BACKEND_PORT:-8000}"
echo "React App: http://localhost:${REACT_PORT:-3000}"
echo "Web Interface: http://localhost:${WEB_PORT:-8081}"
echo "Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}"
echo "Grafana: http://localhost:${GRAFANA_PORT:-3001}"
echo ""
echo "Use './logs.sh' to view service logs"
echo "Use './stop.sh' to stop services"