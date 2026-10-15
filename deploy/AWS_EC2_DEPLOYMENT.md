# KneeAI – AWS EC2 Deployment Guide

## Architecture

```
Internet
   │
   ▼
EC2 Instance (t3.xlarge / g4dn.xlarge)
   │
   ├── Port 8000 → Docker: kneeai-backend  (FastAPI + SegResNet)
   └── Port 3000 → Docker: kneeai-frontend (Next.js)
```

---

## Step 1: Launch EC2 Instance

### AWS Console
1. Go to **EC2 → Launch Instance**
2. Choose AMI: **Ubuntu Server 22.04 LTS** (64-bit x86)
3. Instance type: **t3.xlarge** (4 vCPU, 16 GB RAM) — minimum
4. Key pair: create or select an existing `.pem` key
5. Storage: **50 GB gp3**
6. Security Group: create new with these inbound rules:

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| SSH | TCP | 22 | Your IP |
| Custom TCP | TCP | 8000 | 0.0.0.0/0 |
| Custom TCP | TCP | 3000 | 0.0.0.0/0 |

7. Click **Launch Instance**
8. Note the **Public IPv4 address** (e.g., `54.123.45.67`)

### AWS CLI Alternative
```bash
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-XXXXXXXX \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=kneeai-prod}]'
```

---

## Step 2: SSH Into Instance

```powershell
# Windows PowerShell
ssh -i "path\to\your-key.pem" ubuntu@54.123.45.67
```

```bash
# macOS / Linux
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@54.123.45.67
```

---

## Step 3: Run Bootstrap Script (One-Time Setup)

```bash
# On the EC2 instance:
curl -sSL https://raw.githubusercontent.com/shamsundarsk/MR/main/deploy/ec2_setup.sh | bash
```

**Or clone manually:**
```bash
git clone https://github.com/shamsundarsk/MR.git ~/kneeai
cd ~/kneeai
chmod +x deploy/ec2_setup.sh
./deploy/ec2_setup.sh
```

The script will:
1. Install Docker + Docker Compose
2. Clone the repository
3. Auto-generate a secure JWT secret
4. Build Docker images (5–10 min first time)
5. Start both services
6. Run health checks

---

## Step 4: Verify Deployment

```bash
# Check containers are running
docker compose -f docker-compose.prod.yml ps

# Test API
curl http://localhost:8000/
curl http://localhost:8000/dataset/summary

# View logs
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f frontend
```

**From your laptop (replace IP):**
- Backend API: http://54.123.45.67:8000
- Swagger Docs: http://54.123.45.67:8000/docs  ← interactive API explorer
- Frontend UI:  http://54.123.45.67:3000

---

## Step 5: Upload Model & Data (Optional)

If the EC2 instance doesn't have the trained model checkpoint:

```bash
# Copy model checkpoint from local machine to EC2
scp -i your-key.pem \
    models/segresnet_knee/best_model.pth \
    ubuntu@54.123.45.67:~/kneeai/models/segresnet_knee/

# Copy demo artifacts
scp -i your-key.pem -r \
    artifacts/cases/demo-phantom-right \
    ubuntu@54.123.45.67:~/kneeai/artifacts/cases/
```

---

## Step 6: Update / Redeploy

After pushing new code to GitHub:

```bash
ssh -i your-key.pem ubuntu@54.123.45.67
cd ~/kneeai
./deploy/ec2_deploy.sh
```

---

## Useful Management Commands

```bash
# View running containers
docker compose -f docker-compose.prod.yml ps

# Stop everything
docker compose -f docker-compose.prod.yml down

# Restart a single service
docker compose -f docker-compose.prod.yml restart backend

# View last 50 lines of logs
docker compose -f docker-compose.prod.yml logs --tail=50 backend

# Enter backend container for debugging
docker exec -it kneeai-backend bash

# Monitor resource usage
docker stats

# Check disk space
df -h
```

---

## Troubleshooting

### Port not accessible
- Check EC2 Security Group inbound rules for ports 8000 and 3000
- Make sure instance is in "running" state

### Backend starts but returns 500
```bash
docker compose -f docker-compose.prod.yml logs backend | tail -50
```
- Usually missing model file or data directory

### Out of memory
- Upgrade to t3.xlarge (16 GB) or larger
- SegResNet inference requires ~4 GB RAM minimum

### Docker build fails
```bash
docker compose -f docker-compose.prod.yml build --no-cache backend
```

---

## Cost Estimate (AWS us-east-1)

| Resource | Type | Monthly Cost |
|----------|------|-------------|
| EC2 Instance | t3.xlarge | ~$120 |
| EBS Storage (50 GB gp3) | Storage | ~$4 |
| Data Transfer | 10 GB/month | ~$1 |
| **Total** | | **~$125/month** |

Stop the instance when not in use to save costs:
```bash
aws ec2 stop-instances --instance-ids i-XXXXXXXXXXXXXXXXX
```
