# Quick Start Guide - React Frontend

## 🚀 Get Running in 3 Minutes

### Prerequisites Check
```bash
# Check Node.js (need 16+)
node --version

# Check npm
npm --version

# Backend must be running
curl http://localhost:5000/health
```

---

## Step 1: Install (1 minute)

```bash
cd esports-frontend
npm install
```

---

## Step 2: Start (30 seconds)

```bash
npm run dev
```

The app will open automatically at `http://localhost:3000`

---

## Step 3: Test (1 minute)

1. Click **"Load Example Match"** button
2. Keep default settings
3. Click **"Analyze Match"**
4. View results in ~5 seconds!

---

## 🎯 Quick Tips

### Loading Example Data
Click the sparkle button "Load Example Match" to see a pre-filled Dota 2 match.

### Trying Different Analysis
Click the cards in "Analysis Configuration":
- **Complete**: Full analysis with all insights
- **Summary**: Just the match overview
- **Tactics**: Strategic recommendations
- **Pivots**: Key turning points

### Changing Summary Style
Under "Summary Style" dropdown:
- **Analytical**: Technical breakdown
- **Narrative**: Story-like recap
- **Bullet**: Quick points
- **Tactical**: Strategy-focused

---

## 🐛 Troubleshooting

### "Failed to fetch"
**Problem**: Can't connect to backend  
**Fix**: Start the Flask backend first:
```bash
cd esports-backend
python esports_backend.py
```

### Port 3000 in use
**Problem**: Port already taken  
**Fix**: Change port in `vite.config.js`:
```javascript
server: {
  port: 3001,  // Change this
}
```

### Blank page
**Problem**: JavaScript error  
**Fix**: Open browser console (F12) and check for errors

---

## 📋 Essential Commands

```bash
# Development
npm run dev          # Start dev server

# Production
npm run build        # Build for production
npm run preview      # Preview production build

# Other
npm run lint         # Check code quality
```

---

## 🎨 Quick Customization

### Change Theme Color
Edit `src/App.jsx`, find `:root` and change:
```css
--accent-primary: #00f0ff;  /* Your color here */
```

### Change Backend URL
Edit `vite.config.js`:
```javascript
proxy: {
  '/api': {
    target: 'http://your-backend-url:5000',
  }
}
```

---

## 📱 What You'll See

### Header
- **TACTICAL COMMAND** - Animated title with glow effect
- Subtitle: "ESports Strategy Analyzer"

### Left Panel - Match Data
- Team names, winner, duration
- Compositions (hero/champion picks)
- Events timeline
- Commentary and statistics

### Right Panel - Analysis Config
- 4 analysis type cards
- Style/focus/length dropdowns
- Team selection for recommendations
- Analyze button

### Results (after analysis)
- Tabbed interface
- Summary / Tactics / Pivots
- Metadata showing settings used
- Formatted text output

---

## 🎮 Example Workflow

1. **Click "Load Example Match"** → Pre-fills Liquid vs EG
2. **Select "Complete"** analysis type
3. **Choose "Tactical"** style
4. **Set focus to "Objectives"**
5. **Click "Analyze Match"**
6. **Wait 5-10 seconds**
7. **Review tabs**: Summary → Tactics → Pivots

---

## ✅ Success Indicators

You'll know it's working when:
- ✅ Page loads with dark tactical theme
- ✅ Forms are editable
- ✅ Example button loads data
- ✅ Analyze button is enabled (not grayed out)
- ✅ Results appear after clicking analyze

---

## 🔗 Important URLs

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:5000
- **API Health**: http://localhost:5000/health
- **API Config**: http://localhost:5000/api/config

---

## 📞 Need Help?

1. **Check backend**: Visit http://localhost:5000/health
2. **Check console**: Press F12 in browser
3. **Read full docs**: See `FRONTEND_README.md`
4. **Test backend**: Use Postman or curl

---

## 🎯 Common Mistakes

❌ **Forgot to start backend**  
→ Start Flask backend first!

❌ **Wrong Node version**  
→ Need Node 16 or higher

❌ **Missing dependencies**  
→ Run `npm install` first

❌ **Firewall blocking**  
→ Allow localhost:3000 and :5000

---

## 🚀 Next Steps

After basic setup works:

1. **Try different matches** - Enter your own data
2. **Experiment with styles** - See how outputs differ
3. **Compare analyses** - Run same match with different settings
4. **Test all modes** - Try Summary, Tactics, Pivots separately
5. **Customize theme** - Change colors to your preference

---

**You're ready to analyze matches! 🎮**

*Need more details? Check FRONTEND_README.md*
