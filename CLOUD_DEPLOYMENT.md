# Terra Constellata Cloud Deployment Guide

This guide provides instructions for deploying Terra Constellata on open-source cloud platforms like DigitalOcean and Linode.

## Prerequisites

- Docker and Docker Compose installed on your local machine
- Git repository access
- Cloud provider account (DigitalOcean or Linode)

## DigitalOcean Deployment (Docker-based)

### 1. Create a Droplet

1. Log in to your DigitalOcean account
2. Click "Create" > "Droplets"
3. Choose Ubuntu 22.04 LTS
4. Select a plan (at least 2GB RAM, 1 CPU for basic deployment)
5. Choose a datacenter region
6. Add your SSH keys
7. Create the droplet

### 2. Initial Server Setup

```bash
# SSH into your droplet
ssh root@your_droplet_ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
sudo apt install -y docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group (optional)
sudo usermod -aG docker $USER
```

### 3. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/your-repo/terra-constellata.git
cd terra-constellata

# Copy environment file
cp .env.example .env

# Edit environment variables
nano .env
```

### 4. Configure Environment Variables

Update the `.env` file with secure passwords and appropriate settings:

```env
# Database Configuration
POSTGRES_DB=terra_constellata
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_PORT=5432

# ArangoDB Configuration
ARANGO_ROOT_PASSWORD=your_secure_arango_password

# Other configurations...
```

### 5. Deploy with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 6. Configure Firewall

```bash
# Allow necessary ports
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 3000/tcp  # React app
sudo ufw allow 8000/tcp  # Backend API
sudo ufw allow 8080/tcp  # A2A Server
sudo ufw allow 8081/tcp  # Web interface
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw allow 3001/tcp  # Grafana

# Enable firewall
sudo ufw enable
```

### 7. Set up SSL (Optional but Recommended)

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d yourdomain.com
```

## Linode Deployment (Kubernetes-based)

### 1. Create a Linode

1. Log in to your Linode account
2. Click "Create" > "Linode"
3. Choose Ubuntu 22.04 LTS
4. Select a plan (at least 4GB RAM for Kubernetes)
5. Choose a region
6. Add SSH keys
7. Create the Linode

### 2. Install Kubernetes

```bash
# SSH into your Linode
ssh root@your_linode_ip

# Install k3s (lightweight Kubernetes)
curl -sfL https://get.k3s.io | sh -

# Check status
sudo systemctl status k3s
```

### 3. Install kubectl and Helm

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### 4. Deploy to Kubernetes

```bash
# Clone repository
git clone https://github.com/your-repo/terra-constellata.git
cd terra-constellata

# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods
kubectl get services
```

### 5. Configure Ingress

```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx

# Apply ingress rules
kubectl apply -f k8s/ingress.yml
```

## Monitoring and Maintenance

### Access Monitoring Tools

- **Grafana**: http://your-server-ip:3001 (admin/admin)
- **Prometheus**: http://your-server-ip:9090

### Backup Strategy

```bash
# Backup databases
docker exec terra-postgis pg_dump -U postgres terra_constellata > backup.sql
docker exec terra-arangodb arangodump --output-directory /tmp/arango-backup

# Copy backups to local machine
docker cp terra-postgis:/backup.sql ./backups/
docker cp terra-arangodb:/tmp/arango-backup ./backups/
```

### Scaling

For production deployments, consider:
- Load balancer for multiple instances
- Database replication
- Redis for caching
- CDN for static assets

## Troubleshooting

### Common Issues

1. **Port conflicts**: Check if ports are already in use
2. **Memory issues**: Monitor resource usage with `docker stats`
3. **Database connections**: Verify environment variables
4. **Firewall rules**: Ensure ports are open

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
```

## Security Considerations

- Change default passwords
- Use environment variables for secrets
- Enable SSL/TLS
- Regularly update Docker images
- Monitor for security vulnerabilities
- Implement proper backup and recovery procedures