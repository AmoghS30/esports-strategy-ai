# ESports Strategy Analyzer - React Frontend

A modern, tactical-themed React frontend for the ESports Strategy Analyzer API. Built with React, Vite, and styled with a distinctive cyberpunk/tactical aesthetic.

![ESports Analyzer](https://img.shields.io/badge/React-18.2-blue) ![Vite](https://img.shields.io/badge/Vite-5.0-purple) ![Status](https://img.shields.io/badge/Status-Ready-green)

## 🎨 Design Philosophy

This frontend features a **tactical command center** aesthetic with:
- **Custom Typography**: Orbitron (display) + Space Mono (body) for a technical, futuristic feel
- **Dark Theme**: Deep space blues with cyan and magenta accents
- **Animated Elements**: Glowing effects, smooth transitions, and micro-interactions
- **Grid Scan Effects**: Subtle scan lines for that tactical display feeling
- **Card-Based Layout**: Organized panels with gradient borders and hover effects

## ✨ Features

### 🎯 Analysis Modes
- **Complete Analysis** - Full breakdown with summary, tactics, and pivots
- **Summary Only** - Quick match overview
- **Tactical Recommendations** - Strategic improvement suggestions
- **Turning Points** - Key moments that decided the match

### ⚙️ Customization Options
- **4 Summary Styles**: Analytical, Narrative, Bullet Points, Tactical
- **5 Focus Areas**: Team Fights, Economy, Objectives, Composition, Macro
- **3 Length Options**: Short, Medium, Long
- **Team Selection**: Get recommendations for either team

### 🎮 User Experience
- **Example Data Loading** - One-click example for quick testing
- **Real-time Validation** - Form validation before submission
- **Loading States** - Animated spinners during API calls
- **Error Handling** - Clear error messages
- **Tabbed Results** - Easy navigation between analysis sections
- **Responsive Design** - Works on desktop and tablet

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm/yarn
- Backend API running on `http://localhost:5000`

### Installation

1. **Install dependencies**:
```bash
npm install
# or
yarn install
```

2. **Start development server**:
```bash
npm run dev
# or
yarn dev
```

3. **Open browser**:
The app will automatically open at `http://localhost:3000`

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
esports-frontend/
├── src/
│   ├── App.jsx              # Main application component
│   └── main.jsx             # React entry point
├── public/                   # Static assets
├── index.html               # HTML template
├── package.json             # Dependencies
├── vite.config.js           # Vite configuration
└── README.md               # This file
```

## 🔌 API Integration

The frontend connects to the Flask backend API at `http://localhost:5000`.

### API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/summarize` | POST | Generate match summary |
| `/api/recommendations` | POST | Get tactical recommendations |
| `/api/turning-points` | POST | Identify key moments |
| `/api/analyze` | POST | Complete match analysis |

### CORS Configuration

The backend must have CORS enabled for the frontend to work. The Flask backend includes CORS support via `flask-cors`.

### Proxy Setup

Vite is configured to proxy API requests:
```javascript
proxy: {
  '/api': {
    target: 'http://localhost:5000',
    changeOrigin: true,
  }
}
```

## 🎨 Styling Architecture

### Design System

**Colors:**
```css
--bg-primary: #0a0e1a      /* Deep space blue */
--bg-secondary: #12182b    /* Slightly lighter */
--bg-tertiary: #1a2235     /* Panel backgrounds */
--accent-primary: #00f0ff  /* Cyan - primary actions */
--accent-secondary: #ff3366 /* Magenta - alerts */
--accent-tertiary: #ffaa00  /* Amber - warnings */
```

**Typography:**
- **Display**: Orbitron (900 weight) - Headers and titles
- **Body**: Space Mono - All content and forms
- **Sizing**: 0.75rem - 4rem scale

**Spacing:**
- Base unit: 0.5rem (8px)
- Scale: 1rem, 1.5rem, 2rem, 3rem

### Component Patterns

**Panels:**
- Dark background with glowing borders
- Animated top border on hover
- Consistent padding (2rem)

**Forms:**
- Dark inputs with cyan focus rings
- Uppercase labels with letter spacing
- Full-width responsive

**Buttons:**
- Gradient primary buttons with glow
- Outlined secondary buttons
- Icon + text combinations

**Animations:**
- Glow pulse on title (3s loop)
- Slide-up on results (0.5s)
- Smooth transitions (0.3s)

## 🎯 Component Breakdown

### Main App Component (`App.jsx`)

**State Management:**
- `matchData` - Form inputs for match information
- `analysisType` - Selected analysis mode
- `style/focus/length` - Configuration options
- `loading` - API request state
- `results` - API response data
- `error` - Error messages
- `activeTab` - Current results tab

**Key Functions:**
- `formatMatchData()` - Converts form data to API format
- `analyzeMatch()` - Makes API request
- `loadExample()` - Loads sample data

**Sections:**
1. **Header** - Animated title and subtitle
2. **Match Data Panel** - Form for entering match information
3. **Configuration Panel** - Analysis settings and options
4. **Results Panel** - Tabbed display of analysis results

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```env
VITE_API_BASE_URL=http://localhost:5000
```

### Vite Configuration

**Port**: 3000 (default)  
**Proxy**: API requests to backend  
**Hot Reload**: Enabled  
**Build Output**: `dist/`

## 📱 Responsive Design

### Breakpoints

- **Desktop**: 1024px+
- **Tablet**: 640px - 1024px
- **Mobile**: < 640px

### Responsive Features

- **Grid Layout**: 2-column → 1-column on tablet
- **Form Grid**: 2-column → 1-column on mobile
- **Tab Navigation**: Horizontal scroll on mobile
- **Font Scaling**: Responsive text sizes

## 🎮 Usage Guide

### Step 1: Enter Match Data

**Required Fields:**
- Team A name
- Team B name
- Winner
- Duration (optional but recommended)

**Optional Fields:**
- Team compositions
- Match events (timestamped)
- Commentary
- Statistics

**Quick Tip**: Click "Load Example Match" to see how it works!

### Step 2: Configure Analysis

1. **Choose Analysis Type**:
   - Complete (all insights)
   - Summary (overview only)
   - Tactics (recommendations)
   - Pivots (key moments)

2. **Select Style** (for summary):
   - Analytical (detailed breakdown)
   - Narrative (story-telling)
   - Bullet (quick points)
   - Tactical (strategic focus)

3. **Pick Focus Area**:
   - Team Fights
   - Economy
   - Objectives
   - Composition
   - Macro Strategy

4. **Choose Team** (for recommendations):
   - Team A or Team B

### Step 3: Analyze

Click "Analyze Match" and wait for results (usually 2-10 seconds depending on backend).

### Step 4: Review Results

Navigate between tabs:
- **Summary**: Match overview
- **Tactics**: Strategic recommendations
- **Pivots**: Critical turning points

## 🐛 Troubleshooting

### Common Issues

**Problem**: "Failed to fetch" error  
**Solution**: Ensure backend is running on `http://localhost:5000`

**Problem**: CORS errors  
**Solution**: Backend must have CORS enabled (included in Flask backend)

**Problem**: Styles not loading  
**Solution**: Hard refresh (Ctrl+Shift+R) or clear browser cache

**Problem**: Port 3000 already in use  
**Solution**: Change port in `vite.config.js` or stop other process

### Backend Connection

Test backend connectivity:
```bash
curl http://localhost:5000/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

## 🚀 Deployment

### Option 1: Static Hosting (Vercel/Netlify)

```bash
# Build
npm run build

# Deploy dist/ folder
```

**Important**: Update API base URL for production

### Option 2: Docker

Create `Dockerfile`:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "dist", "-l", "3000"]
EXPOSE 3000
```

Build and run:
```bash
docker build -t esports-frontend .
docker run -p 3000:3000 esports-frontend
```

### Option 3: Serve with Backend

Use NGINX to serve both:
```nginx
server {
    listen 80;
    
    # Frontend
    location / {
        root /path/to/dist;
        try_files $uri /index.html;
    }
    
    # Backend API
    location /api {
        proxy_pass http://localhost:5000;
    }
}
```

## 📦 Dependencies

### Production
- **react**: ^18.2.0 - UI framework
- **react-dom**: ^18.2.0 - DOM rendering
- **lucide-react**: ^0.294.0 - Icon library

### Development
- **vite**: ^5.0.8 - Build tool
- **@vitejs/plugin-react**: ^4.2.1 - React plugin
- **eslint**: ^8.55.0 - Linting

## 🎨 Customization

### Change Colors

Edit CSS variables in `App.jsx`:
```css
:root {
  --accent-primary: #00f0ff;  /* Change to your color */
  --accent-secondary: #ff3366;
  --bg-primary: #0a0e1a;
}
```

### Change Fonts

Update Google Fonts import:
```css
@import url('https://fonts.googleapis.com/css2?family=YourFont&display=swap');
```

Then update font-family:
```css
font-family: 'YourFont', sans-serif;
```

### Add Features

The component is modular - easy to extend:
- Add new analysis types
- Create additional tabs
- Implement saved matches
- Add comparison features

## 📊 Performance

### Optimization Tips

1. **Code Splitting**: Already configured with Vite
2. **Lazy Loading**: Can add React.lazy() for routes
3. **Memoization**: Use React.memo() for expensive renders
4. **Debouncing**: Add debounce for form inputs

### Build Size

Typical production build:
- **Vendor**: ~150KB (React + deps)
- **App**: ~30-50KB (compressed)
- **Total**: ~180-200KB gzipped

## 🔐 Security

### Best Practices

- Never commit API keys
- Sanitize user inputs
- Use HTTPS in production
- Implement rate limiting
- Validate API responses

## 📄 License

Educational/Academic use for ESports Strategy Analyzer project.

## 🙏 Credits

- **Design**: Custom tactical/cyberpunk theme
- **Icons**: Lucide React
- **Fonts**: Google Fonts (Orbitron, Space Mono)
- **Build Tool**: Vite

## 🆘 Support

### Getting Help

1. Check backend is running: `http://localhost:5000/health`
2. Check browser console for errors (F12)
3. Review backend logs
4. Test API with Postman/curl first

### Next Steps

- Connect to deployed backend
- Add user authentication
- Implement match history
- Add data visualization
- Create mobile app version

---

**Built with React + Vite**  
*Optimized for modern browsers*  
*Tactical design for tactical analysis*

**Last Updated**: November 2024  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
