#!/bin/bash

# Terra Constellata Backend Logs Script

SERVICE=${1:-all}

echo "Showing logs for Terra Constellata backend services..."

if [ "$SERVICE" = "all" ]; then
    docker-compose logs -f
elif [ "$SERVICE" = "postgres" ]; then
    docker-compose logs -f postgres
elif [ "$SERVICE" = "arangodb" ]; then
    docker-compose logs -f arangodb
elif [ "$SERVICE" = "a2a-server" ]; then
    docker-compose logs -f a2a-server
elif [ "$SERVICE" = "backend" ]; then
    docker-compose logs -f backend
elif [ "$SERVICE" = "react-app" ]; then
    docker-compose logs -f react-app
elif [ "$SERVICE" = "web" ]; then
    docker-compose logs -f web
elif [ "$SERVICE" = "prometheus" ]; then
    docker-compose logs -f prometheus
elif [ "$SERVICE" = "grafana" ]; then
    docker-compose logs -f grafana
else
    echo "Usage: $0 [all|postgres|arangodb|a2a-server|backend|react-app|web|prometheus|grafana]"
    echo "Showing all logs by default..."
    docker-compose logs -f
fi