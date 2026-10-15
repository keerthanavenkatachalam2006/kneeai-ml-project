# AWS EC2 Security Group – Required Inbound Rules

Configure these rules in the EC2 Security Group attached to your instance.
**Source can be `0.0.0.0/0` for demo; restrict to your IP for real use.**

| Rule | Type | Protocol | Port Range | Source | Purpose |
|------|------|----------|------------|--------|---------|
| 1 | SSH | TCP | 22 | Your IP | Admin access |
| 2 | Custom TCP | TCP | 8000 | 0.0.0.0/0 | Backend API / Swagger UI |
| 3 | Custom TCP | TCP | 3000 | 0.0.0.0/0 | Next.js Frontend |
| 4 | HTTP | TCP | 80 | 0.0.0.0/0 | (Optional) Nginx proxy |
| 5 | HTTPS | TCP | 443 | 0.0.0.0/0 | (Optional) TLS |

## Quick AWS CLI Commands

```bash
# Replace sg-XXXXXXXX with your security group ID

# Allow SSH (replace YOUR_IP)
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXXXXX \
    --protocol tcp --port 22 \
    --cidr YOUR_IP/32

# Allow Backend API
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXXXXX \
    --protocol tcp --port 8000 \
    --cidr 0.0.0.0/0

# Allow Frontend UI
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXXXXX \
    --protocol tcp --port 3000 \
    --cidr 0.0.0.0/0
```

## Recommended Instance Specs

| Tier | Instance | vCPU | RAM | Notes |
|------|----------|------|-----|-------|
| Minimum (demo) | t3.large | 2 | 8 GB | No GPU; CPU inference only |
| Recommended | t3.xlarge | 4 | 16 GB | Comfortable for inference |
| GPU (faster) | g4dn.xlarge | 4 | 16 GB | NVIDIA T4, 10x faster |

- **AMI**: Amazon Linux 2023 or Ubuntu 22.04 LTS
- **Storage**: 50 GB gp3 EBS (data + models + Docker layers)
- **Key pair**: Required for SSH access
