// frontend/src/App.jsx
import { useState, useEffect } from 'react';
import { Gamepad2, TrendingUp, AlertCircle } from 'lucide-react';
import MatchUpload from './components/MatchUpload';
import AdvancedControls from './Components/AdvancedControls';
import ResultsDisplay from './components/ResultsDisplay';
import MatchPreview from './components/MatchPreview';
import { apiService } from './services/api';
import './App.css';

function App() {
  const [matchData, setMatchData] = useState(null);
  const [focusTeam, setFocusTeam] = useState('team_a');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  
  const [promptControls, setPromptControls] = useState({
    length: 'medium',
    focus: ['overall'],
    style: 'analytical'
  });

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      await apiService.healthCheck();
      setApiStatus('connected');
    } catch (err) {
      setApiStatus('disconnected');
      setError('Cannot connect to backend. Make sure the server is running on port 8000.');
    }
  };

  const handleMatchUpload = (data) => {
    setMatchData(data);
    setResult(null);
    setError(null);
  };

  const loadSampleData = () => {
    const sample = {
      match_id: "LOL_WORLDS_2024_001",
      teams: {
        team_a: "T1",
        team_b: "JDG"
      },
      hero_picks: {
        team_a: ["Aatrox", "Lee Sin", "Orianna", "Jinx", "Thresh"],
        team_b: ["Gnar", "Viego", "Ahri", "Aphelios", "Nautilus"]
      },
      events: [
        { time: "2:45", type: "first_blood", team: "team_a", details: "Lee Sin ganks mid lane" },
        { time: "6:30", type: "dragon", team: "team_b", details: "Cloud Dragon secured" },
        { time: "12:20", type: "first_tower", team: "team_a", details: "Top tower falls" },
        { time: "18:45", type: "baron", team: "team_a", details: "Baron secured after team fight" },
        { time: "23:10", type: "team_fight", winner: "team_a", details: "Ace near dragon pit - 5-1" },
        { time: "25:40", type: "victory", team: "team_a", details: "Nexus destroyed" }
      ],
      commentary: `T1 came out strong with an aggressive early game strategy, securing first blood at 2:45 when Lee Sin executed a perfect gank in mid lane. JDG responded with superior objective control, taking the first dragon at 6:30. The match's turning point came at 18:45 when T1 secured Baron Nashor after winning a crucial team fight. This Baron buff enabled them to break JDG's defensive line and take multiple towers. At 23:10, T1 caught JDG rotating to dragon and won a decisive 5-1 team fight that effectively ended the game. Faker's Orianna delivered a game-winning shockwave that caught three members of JDG, allowing T1 to push for the win. T1's superior team fighting and objective control proved decisive in this match.`
    };
    setMatchData(sample);
    setResult(null);
    setError(null);
  };

  const analyzeMatch = async () => {
    if (!matchData) {
      setError('Please upload match data first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await apiService.analyzeMatch(
        matchData,
        focusTeam,
        promptControls
      );
      setResult(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <Gamepad2 size={40} />
            <h1>ESports Strategy AI</h1>
          </div>
          <p className="tagline">Comprehensive AI-Powered Match Analysis</p>
          <p className="subtitle">Get Overview • Turning Points • Tactical Recommendations in One Report</p>
          
          <div className={`api-status ${apiStatus}`}>
            <div className="status-dot"></div>
            <span>
              {apiStatus === 'connected' ? 'Connected' : 
               apiStatus === 'checking' ? 'Checking...' : 
               'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          
          {error && (
            <div className="alert alert-error">
              <AlertCircle size={20} />
              <span>{error}</span>
              <button onClick={() => setError(null)}>×</button>
            </div>
          )}

          <section className="card">
            <div className="card-header">
              <TrendingUp size={24} />
              <h2>Step 1: Load Match Data</h2>
            </div>
            <div className="card-body">
              <MatchUpload 
                onMatchUpload={handleMatchUpload}
                onLoadSample={loadSampleData}
              />
              {matchData && <MatchPreview matchData={matchData} />}
            </div>
          </section>

          {matchData && (
            <section className="card">
              <div className="card-header">
                <TrendingUp size={24} />
                <h2>Step 2: Configure Analysis & Generate Report</h2>
              </div>
              <div className="card-body">
                
                {/* Team Selection */}
                <div className="config-section">
                  <label className="config-label">Select Team for Tactical Recommendations</label>
                  <div className="team-selector">
                    <button
                      className={`team-btn ${focusTeam === 'team_a' ? 'active' : ''}`}
                      onClick={() => setFocusTeam('team_a')}
                    >
                      {matchData.teams.team_a}
                    </button>
                    <button
                      className={`team-btn ${focusTeam === 'team_b' ? 'active' : ''}`}
                      onClick={() => setFocusTeam('team_b')}
                    >
                      {matchData.teams.team_b}
                    </button>
                  </div>
                </div>

                {/* Advanced Controls */}
                <AdvancedControls
                  controls={promptControls}
                  setControls={setPromptControls}
                />

                {/* Analyze Button */}
                <button 
                  className="btn btn-analyze"
                  onClick={analyzeMatch}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <div className="btn-spinner"></div>
                      Analyzing Match...
                    </>
                  ) : (
                    <>
                      🚀 Generate Comprehensive Analysis
                    </>
                  )}
                </button>
              </div>
            </section>
          )}

          {result && (
            <section className="card">
              <ResultsDisplay 
                result={result}
                promptControls={promptControls}
              />
            </section>
          )}

          {loading && (
            <div className="loading-container">
              <div className="loading-spinner"></div>
              <p>Generating comprehensive match analysis...</p>
              <p className="loading-subtext">
                Including: Overview • Turning Points • Tactical Recommendations
              </p>
              <p className="loading-subtext">
                Settings: {promptControls.length} length • {promptControls.style} style
              </p>
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>Built with ❤️ using Vite + React + FastAPI + Groq AI</p>
      </footer>
    </div>
  );
}

export default App;