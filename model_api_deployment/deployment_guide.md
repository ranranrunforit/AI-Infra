# Cloud Deployment Guide

## Option 1: AWS EC2

### 1. Launch EC2 Instance

```bash
# Launch t3.medium instance with Ubuntu 22.04
# - 2 vCPUs, 4GB RAM
# - 30GB EBS volume
# - Allow inbound traffic on ports 22 (SSH) and 5000 (HTTP)
```

### 2. Connect to Instance

```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

### 3. Install Docker

```bash
# Update packages
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Log out and back in for group changes to take effect
exit
```

### 4. Deploy Application

```bash
# Clone your repository
git clone https://github.com/yourusername/model-api-deployment.git
cd model-api-deployment

# Build Docker image
docker build -f docker/Dockerfile -t model-api:v1.0 .

# Run container
docker run -d \
  -p 5000:5000 \
  --name model-api \
  --restart unless-stopped \
  model-api:v1.0

# Check logs
docker logs -f model-api
```

### 5. Test Deployment

```bash
# From your local machine
curl http://your-instance-ip:5000/health
curl http://your-instance-ip:5000/info
```

### 6. Set Up Monitoring (Optional)

```bash
# View container stats
docker stats model-api

# Set up CloudWatch logging
# Follow AWS documentation for container logging
```

## Option 2: GCP Compute Engine

Similar steps but use:
- Google Cloud Console to launch VM
- gcloud CLI for management
- Similar Docker setup

## Option 3: Azure VM

Similar steps but use:
- Azure Portal to launch VM
- Azure CLI for management
- Similar Docker setup

## Security Checklist

- [ ] SSH access restricted to your IP
- [ ] API endpoint accessible on port 5000
- [ ] HTTPS not configured (add nginx/certbot later)
- [ ] No sensitive data in logs
- [ ] Environment variables properly set

## Troubleshooting

**Container won't start:**
```bash
docker logs model-api
```

**Out of memory:**
```bash
# Check memory
free -h
# Consider using mobilenet_v2 (lighter model)
```

**Port already in use:**
```bash
# Stop conflicting service
sudo lsof -i :5000
```
