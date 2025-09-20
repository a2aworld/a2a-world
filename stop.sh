#!/bin/bash

# Terra Constellata Backend Stop Script

echo "Stopping Terra Constellata backend services..."

# Stop services
docker-compose down

echo "Services stopped."
echo "Data volumes preserved. Use 'docker-compose down -v' to remove volumes."