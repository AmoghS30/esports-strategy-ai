# Complete Deployment Guide

## 🎯 Full Stack Deployment

This guide covers deploying both the Flask backend and React frontend together.

---

## 📦 What You Have

### Backend (esports-backend/)
- Flask API with ML inference
- Supports GPU/CPU
- Port 5000

### Frontend (esports-frontend/)
- React + Vite SPA
- Modern UI
- Port 3000

---

## 🚀 Local Development Setup

### Option 1: Quick Start (Both Services)

**Terminal 1 - Backend:**
```bash
cd esports-backend
bash start_server.sh
```

**Terminal 2 - Frontend:**
```bash
cd esports-frontend
npm install
npm run dev
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:5000
- Health Check: http://localhost:5000/health

### Option 2: Production Mode

**Backend:**
```bash
cd esports-backend
source venv/bin/activate
gunicorn -w 4 -b 0.0.0.0:5000 esports_backend:app
```

**Frontend:**
```bash
cd esports-frontend
npm run build
npm run preview
```

---

## 🐳 Docker Deployment

### Backend Dockerfile

Create `esports-backend/Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY esports_backend.py .
COPY models/ models/

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "esports_backend:app"]
```

### Frontend Dockerfile

Create `esports-frontend/Dockerfile`:
```dockerfile
FROM node:18-alpine AS build

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Frontend nginx.conf

Create `esports-frontend/nginx.conf`:
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Frontend
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API to backend
    location /api {
        proxy_pass http://backend:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  backend:
    build: ./esports-backend
    ports:
      - "5000:5000"
    volumes:
      - ./esports-backend/models:/app/models
    environment:
      - MODEL_DIR=/app/models/esports-gpu-fast_final
    restart: unless-stopped

  frontend:
    build: ./esports-frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
```

**Run:**
```bash
docker-compose up -d
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### Option A: EC2 Instance

**1. Launch EC2 (t3.medium or g5.2xlarge for GPU)**

**2. Install dependencies:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python
sudo apt install -y python3-pip python3-venv

# Install NGINX
sudo apt install -y nginx
```

**3. Deploy Backend:**
```bash
cd /var/www
sudo git clone <your-repo>
cd esports-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Setup Systemd Service:**

Create `/etc/systemd/system/esports-backend.service`:
```ini
[Unit]
Description=ESports Backend API
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/esports-backend
Environment="PATH=/var/www/esports-backend/venv/bin"
ExecStart=/var/www/esports-backend/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 esports_backend:app

[Install]
WantedBy=multi-user.target
```

**5. Deploy Frontend:**
```bash
cd /var/www/esports-frontend
npm install
npm run build
```

**6. Configure NGINX:**

Create `/etc/nginx/sites-available/esports`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /var/www/esports-frontend/dist;
        try_files $uri /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend health
    location /health {
        proxy_pass http://127.0.0.1:5000;
    }
}
```

**7. Enable and start:**
```bash
sudo ln -s /etc/nginx/sites-available/esports /etc/nginx/sites-enabled/
sudo systemctl enable esports-backend
sudo systemctl start esports-backend
sudo systemctl restart nginx
```

#### Option B: ECS with Fargate

**1. Push images to ECR:**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Build and push backend
cd esports-backend
docker build -t esports-backend .
docker tag esports-backend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/esports-backend:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/esports-backend:latest

# Build and push frontend
cd ../esports-frontend
docker build -t esports-frontend .
docker tag esports-frontend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/esports-frontend:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/esports-frontend:latest
```

**2. Create ECS Task Definition**
**3. Create ECS Service**
**4. Configure Application Load Balancer**

### Vercel (Frontend Only)

**1. Install Vercel CLI:**
```bash
npm i -g vercel
```

**2. Deploy:**
```bash
cd esports-frontend
vercel
```

**3. Configure environment:**
- Set `VITE_API_BASE_URL` to your backend URL

### Netlify (Frontend Only)

**1. Build command:** `npm run build`
**2. Publish directory:** `dist`
**3. Configure redirects:**

Create `esports-frontend/public/_redirects`:
```
/api/* https://your-backend-url.com/api/:splat 200
/* /index.html 200
```

### Heroku (Backend)

**1. Create `Procfile`:**
```
web: gunicorn -w 4 -b 0.0.0.0:$PORT esports_backend:app
```

**2. Deploy:**
```bash
heroku login
heroku create esports-backend
git push heroku main
```

---

## 🔒 Production Checklist

### Backend
- [ ] Environment variables configured
- [ ] Model files accessible
- [ ] CORS properly configured
- [ ] Rate limiting implemented
- [ ] Error logging setup
- [ ] Health checks working
- [ ] SSL certificate installed
- [ ] Firewall configured
- [ ] Backup strategy in place

### Frontend
- [ ] API URL updated for production
- [ ] Environment variables set
- [ ] Build optimization complete
- [ ] CDN configured (optional)
- [ ] Analytics added (optional)
- [ ] Error tracking setup
- [ ] SEO meta tags added
- [ ] Favicon configured

### Security
- [ ] HTTPS enabled
- [ ] API authentication (if needed)
- [ ] Rate limiting
- [ ] Input validation
- [ ] SQL injection protection (N/A)
- [ ] XSS protection
- [ ] CORS properly restricted
- [ ] Security headers configured

---

## 🔧 Configuration

### Backend Environment Variables

**Production `.env`:**
```bash
MODEL_DIR=/app/models/esports-gpu-fast_final
MAX_LENGTH=512
TEMPERATURE=0.7
FLASK_ENV=production
LOG_LEVEL=INFO
```

### Frontend Environment Variables

**Production `.env`:**
```bash
VITE_API_BASE_URL=https://api.yourdomain.com
```

---

## 📊 Monitoring

### Backend Monitoring

**Health Check:**
```bash
curl https://api.yourdomain.com/health
```

**Logs:**
```bash
# Systemd
sudo journalctl -u esports-backend -f

# Docker
docker logs -f <container-id>
```

### Frontend Monitoring

**Check build:**
```bash
npm run build
# Look for errors
```

**Test production:**
```bash
npm run preview
```

---

## 🚨 Troubleshooting

### Backend Issues

**Model not loading:**
```bash
# Check model path
ls -la /path/to/models/esports-gpu-fast_final

# Check permissions
chmod -R 755 /path/to/models
```

**Port conflicts:**
```bash
# Check what's using port 5000
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>
```

**Memory issues:**
```bash
# Check memory
free -h

# Check GPU memory (if applicable)
nvidia-smi
```

### Frontend Issues

**Build failures:**
```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install
npm run build
```

**API connection:**
```bash
# Check backend is accessible
curl https://api.yourdomain.com/health

# Check CORS
curl -H "Origin: https://yourdomain.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS https://api.yourdomain.com/api/summarize
```

---

## 🎯 Performance Optimization

### Backend
- Use Gunicorn with multiple workers
- Enable GPU if available
- Implement caching (Redis)
- Use connection pooling
- Optimize model loading

### Frontend
- Enable gzip compression
- Use CDN for static assets
- Implement lazy loading
- Optimize images
- Code splitting

---

## 📈 Scaling

### Horizontal Scaling (Backend)
```bash
# Run multiple instances
gunicorn -w 8 -b 0.0.0.0:5000 esports_backend:app
```

### Load Balancing
Use NGINX, HAProxy, or AWS ALB to distribute traffic.

### Database (Future)
If adding persistence:
- PostgreSQL for structured data
- Redis for caching
- S3 for model storage

---

## 🔄 CI/CD Pipeline

### GitHub Actions Example

`.github/workflows/deploy.yml`:
```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to server
        run: |
          # Your deployment script

  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and deploy
        run: |
          npm install
          npm run build
          # Deploy to hosting
```

---

## 📝 Maintenance

### Regular Tasks
- Monitor logs daily
- Update dependencies monthly
- Backup models weekly
- Review performance metrics
- Update SSL certificates (auto with Let's Encrypt)

### Updating Application
```bash
# Backend
cd esports-backend
git pull
pip install -r requirements.txt
sudo systemctl restart esports-backend

# Frontend
cd esports-frontend
git pull
npm install
npm run build
# Copy dist/ to nginx root
```

---

## 🎉 Success Criteria

Your deployment is successful when:
- ✅ Frontend loads at your domain
- ✅ Backend health check returns 200
- ✅ Can analyze a test match
- ✅ Results display correctly
- ✅ No console errors
- ✅ Mobile responsive
- ✅ HTTPS enabled
- ✅ Performance acceptable (<3s page load)

---

**Ready for Production! 🚀**

*Choose your deployment strategy and follow the steps above*
