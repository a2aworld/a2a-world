# Terra Constellata Deployment Guide

This document provides comprehensive instructions for deploying the Terra Constellata system locally and on cloud platforms.

## Local Deployment

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/terra-constellata.git
   cd terra-constellata
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

3. **Start services**
   ```bash
   ./start.sh
   ```

4. **Access the application**
   - React App: http://localhost:3000
   - Web Interface: http://localhost:8081
   - Backend API: http://localhost:8000
   - Grafana: http://localhost:3001 (admin/admin)
   - Prometheus: http://localhost:9090

### Services Overview

- **PostgreSQL (PostGIS)**: Spatial database for geographic data
- **ArangoDB**: Multi-model database for knowledge graphs
- **A2A Server**: Agent-to-Agent protocol server
- **Backend**: FastAPI application with all agents and services
- **React App**: Modern web interface
- **Web Interface**: Simple HTML/JS interface
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboard

## Cloud Deployment

See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for detailed cloud deployment instructions.

## Configuration

### Environment Variables

Key configuration options in `.env`:

- Database credentials
- Service ports
- Monitoring settings
- Security configurations

### Networking

All services communicate through the `terra-network` Docker network with proper service discovery.

## Monitoring and Logging

### Monitoring

- **Prometheus**: Collects metrics from all services
- **Grafana**: Provides dashboards for visualization
- Health checks configured for all services

### Logging

- Centralized logging through Docker Compose
- Individual service logs accessible via `./logs.sh <service>`

## Security

### Best Practices

- Change default passwords
- Use strong passwords for production
- Enable SSL/TLS for public deployments
- Regularly update Docker images
- Monitor security vulnerabilities

### Network Security

- Services communicate internally via Docker network
- External access controlled by firewall rules
- API endpoints protected by CORS configuration

## Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check what's using the port
   lsof -i :PORT_NUMBER
   # Change port in .env file
   ```

2. **Memory issues**
   ```bash
   # Monitor resource usage
   docker stats
   # Increase Docker memory limit if needed
   ```

3. **Database connection errors**
   ```bash
   # Check database logs
   ./logs.sh postgres
   ./logs.sh arangodb
   ```

4. **Service startup failures**
   ```bash
   # Check all logs
   ./logs.sh all
   # Restart services
   ./stop.sh
   ./start.sh
   ```

### Health Checks

All services include health checks that can be monitored via:

```bash
# Check service health
curl http://localhost:SERVICE_PORT/health
```

## Development

### Adding New Services

1. Create Dockerfile in appropriate directory
2. Add service configuration to `docker-compose.yml`
3. Update environment variables in `.env`
4. Add health checks and monitoring
5. Update deployment scripts

### Scaling

For production scaling:

- Use Docker Swarm or Kubernetes
- Implement load balancers
- Set up database replication
- Add Redis for caching
- Use CDN for static assets

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
docker exec terra-postgis pg_dump -U postgres terra_constellata > backup.sql

# ArangoDB backup
docker exec terra-arangodb arangodump --output-directory /tmp/backup
```

### Volume Backup

```bash
# Backup Docker volumes
docker run --rm -v terra-constellata_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .
```

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review service logs
3. Verify configuration
4. Check Docker and system resources
5. Consult the project documentation

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Interfaces    │    │    Backend      │
│  (React/Web)    │◄──►│   (FastAPI)     │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐
│   Databases     │    │   A2A Server    │
│ (PostGIS/Arango)│◄──►│                 │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │    Agents       │
│(Prometheus/Grafana)│    │(Sentinel/etc) │
└─────────────────┘    └─────────────────┘
```

This deployment provides a complete, production-ready setup for the Terra Constellata system with monitoring, logging, and security configurations.