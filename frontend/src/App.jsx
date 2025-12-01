import React, { useState } from 'react';
import { Zap, Target, TrendingUp, AlertCircle, Play, ChevronRight, Sparkles, Shield, Trophy, Activity } from 'lucide-react';

const API_BASE_URL = 'http://35.154.3.181:8080';

export default function EsportsAnalyzer() {
  const [matchData, setMatchData] = useState({
    team_a: '',
    team_b: '',
    winner: '',
    duration: '',
    team_a_composition: '',
    team_b_composition: '',
    events: '',
    commentary: '',
    statistics: ''
  });

  const [analysisType, setAnalysisType] = useState('complete');
  const [style, setStyle] = useState('analytical');
  const [focus, setFocus] = useState('team_fights');
  const [length, setLength] = useState('medium');
  const [team, setTeam] = useState('team_a');
  
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('summary');

  const formatMatchData = () => {
    const data = {
      team_a: matchData.team_a,
      team_b: matchData.team_b,
      winner: matchData.winner,
      duration: matchData.duration
    };

    if (matchData.team_a_composition) data.team_a_composition = matchData.team_a_composition;
    if (matchData.team_b_composition) data.team_b_composition = matchData.team_b_composition;
    
    if (matchData.events) {
      data.events = matchData.events.split('\n').filter(e => e.trim());
    }
    
    if (matchData.commentary) data.commentary = matchData.commentary;
    
    if (matchData.statistics) {
      try {
        data.statistics = JSON.parse(matchData.statistics);
      } catch (e) {
        const stats = {};
        matchData.statistics.split('\n').forEach(line => {
          const [key, value] = line.split(':').map(s => s.trim());
          if (key && value) stats[key] = value;
        });
        data.statistics = stats;
      }
    }

    return data;
  };

  const analyzeMatch = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formattedData = formatMatchData();
      let endpoint, payload;

      switch (analysisType) {
        case 'summary':
          endpoint = '/api/summarize';
          payload = { match_data: formattedData, style, focus, length };
          break;
        case 'recommendations':
          endpoint = '/api/recommendations';
          payload = { match_data: formattedData, team, recommendation_depth: 4 };
          break;
        case 'turning-points':
          endpoint = '/api/turning-points';
          payload = { match_data: formattedData };
          break;
        case 'complete':
        default:
          endpoint = '/api/analyze';
          payload = { match_data: formattedData, style, focus, team, recommendation_depth: 3 };
      }

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      setResults(data);
      
      if (analysisType === 'complete') {
        setActiveTab('summary');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadExample = () => {
    setMatchData({
      team_a: 'Team Liquid',
      team_b: 'Evil Geniuses',
      winner: 'Team Liquid',
      duration: '42:15',
      team_a_composition: 'Invoker, Anti-Mage, Lion, Earthshaker, Crystal Maiden',
      team_b_composition: 'Phantom Assassin, Queen of Pain, Rubick, Tidehunter, Ancient Apparition',
      events: `5:30 - Team Liquid secures first blood mid lane
12:00 - Evil Geniuses takes first tower bot lane
18:45 - Major team fight at Roshan pit, Team Liquid wins 4-1
25:30 - Evil Geniuses secures Roshan
32:00 - Team Liquid wins decisive team fight, wipes Evil Geniuses
35:15 - Team Liquid takes mid barracks
42:15 - Team Liquid destroys ancient`,
      commentary: "An intense match between two top-tier teams. Team Liquid's Invoker controlled the mid game with exceptional spell combos, while Evil Geniuses struggled to protect their Phantom Assassin.",
      statistics: `Team Liquid Kills: 45
Evil Geniuses Kills: 32
Team Liquid Gold: 65.2k
Evil Geniuses Gold: 58.1k`
    });
  };

  return (
    <div className="app-container">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Space+Mono:wght@400;700&display=swap');

        :root {
          --bg-primary: #0a0e1a;
          --bg-secondary: #12182b;
          --bg-tertiary: #1a2235;
          --accent-primary: #00f0ff;
          --accent-secondary: #ff3366;
          --accent-tertiary: #ffaa00;
          --text-primary: #e8edf4;
          --text-secondary: #8b98b0;
          --text-muted: #5a6580;
          --border: rgba(0, 240, 255, 0.2);
          --glow: rgba(0, 240, 255, 0.4);
        }

        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          background: var(--bg-primary);
          font-family: 'Space Mono', monospace;
          color: var(--text-primary);
          overflow-x: hidden;
        }

        .app-container {
          min-height: 100vh;
          background: 
            radial-gradient(circle at 20% 20%, rgba(0, 240, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 51, 102, 0.05) 0%, transparent 50%),
            var(--bg-primary);
          padding: 2rem;
          position: relative;
        }

        .app-container::before {
          content: '';
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: 
            repeating-linear-gradient(
              0deg,
              rgba(0, 240, 255, 0.03) 0px,
              transparent 1px,
              transparent 2px,
              rgba(0, 240, 255, 0.03) 3px
            );
          pointer-events: none;
          z-index: 1;
        }

        .content-wrapper {
          max-width: 1400px;
          margin: 0 auto;
          position: relative;
          z-index: 2;
        }

        .header {
          text-align: center;
          margin-bottom: 3rem;
          position: relative;
          padding: 3rem 0;
        }

        .header::after {
          content: '';
          position: absolute;
          bottom: 0;
          left: 50%;
          transform: translateX(-50%);
          width: 300px;
          height: 2px;
          background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
          box-shadow: 0 0 20px var(--glow);
        }

        .title {
          font-family: 'Orbitron', sans-serif;
          font-size: 4rem;
          font-weight: 900;
          letter-spacing: -2px;
          background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-bottom: 0.5rem;
          text-transform: uppercase;
          animation: glow 3s ease-in-out infinite;
        }

        @keyframes glow {
          0%, 100% { filter: drop-shadow(0 0 20px var(--accent-primary)); }
          50% { filter: drop-shadow(0 0 40px var(--accent-primary)); }
        }

        .subtitle {
          font-family: 'Orbitron', sans-serif;
          font-size: 1rem;
          color: var(--text-secondary);
          letter-spacing: 4px;
          text-transform: uppercase;
        }

        .main-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
          margin-bottom: 2rem;
        }

        @media (max-width: 1024px) {
          .main-grid {
            grid-template-columns: 1fr;
          }
        }

        .panel {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 2rem;
          position: relative;
          overflow: hidden;
          transition: all 0.3s ease;
        }

        .panel::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 3px;
          background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
          opacity: 0;
          transition: opacity 0.3s ease;
        }

        .panel:hover::before {
          opacity: 1;
        }

        .panel-header {
          display: flex;
          align-items: center;
          gap: 1rem;
          margin-bottom: 1.5rem;
          font-family: 'Orbitron', sans-serif;
          font-size: 1.25rem;
          font-weight: 700;
          color: var(--accent-primary);
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .panel-header svg {
          width: 24px;
          height: 24px;
        }

        .form-group {
          margin-bottom: 1.5rem;
        }

        .form-label {
          display: block;
          margin-bottom: 0.5rem;
          font-size: 0.875rem;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .form-input, .form-textarea, .form-select {
          width: 100%;
          padding: 0.875rem 1rem;
          background: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: 8px;
          color: var(--text-primary);
          font-family: 'Space Mono', monospace;
          font-size: 0.875rem;
          transition: all 0.3s ease;
        }

        .form-input:focus, .form-textarea:focus, .form-select:focus {
          outline: none;
          border-color: var(--accent-primary);
          box-shadow: 0 0 0 3px rgba(0, 240, 255, 0.1);
        }

        .form-textarea {
          min-height: 100px;
          resize: vertical;
        }

        .form-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1rem;
        }

        @media (max-width: 640px) {
          .form-grid {
            grid-template-columns: 1fr;
          }
        }

        .button-group {
          display: flex;
          gap: 1rem;
          margin-top: 2rem;
        }

        .btn {
          flex: 1;
          padding: 1rem 2rem;
          font-family: 'Orbitron', sans-serif;
          font-size: 0.875rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 1px;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
        }

        .btn-primary {
          background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
          color: var(--bg-primary);
          box-shadow: 0 4px 20px rgba(0, 240, 255, 0.3);
        }

        .btn-primary:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 6px 30px rgba(0, 240, 255, 0.5);
        }

        .btn-secondary {
          background: var(--bg-tertiary);
          color: var(--text-primary);
          border: 1px solid var(--border);
        }

        .btn-secondary:hover:not(:disabled) {
          background: var(--bg-secondary);
          border-color: var(--accent-primary);
        }

        .btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .loading-spinner {
          display: inline-block;
          width: 20px;
          height: 20px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          border-top-color: var(--accent-primary);
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .analysis-type-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
          margin-bottom: 1.5rem;
        }

        .analysis-type-card {
          padding: 1rem;
          background: var(--bg-tertiary);
          border: 2px solid transparent;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.3s ease;
          text-align: center;
        }

        .analysis-type-card.active {
          background: linear-gradient(135deg, rgba(0, 240, 255, 0.1), rgba(255, 51, 102, 0.1));
          border-color: var(--accent-primary);
        }

        .analysis-type-card:hover {
          border-color: var(--accent-primary);
        }

        .analysis-type-icon {
          margin-bottom: 0.5rem;
          color: var(--accent-primary);
        }

        .analysis-type-title {
          font-family: 'Orbitron', sans-serif;
          font-size: 0.875rem;
          font-weight: 700;
          color: var(--text-primary);
          text-transform: uppercase;
        }

        .analysis-type-desc {
          font-size: 0.75rem;
          color: var(--text-muted);
          margin-top: 0.25rem;
        }

        .results-panel {
          grid-column: 1 / -1;
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: 12px;
          overflow: hidden;
          animation: slideUp 0.5s ease;
        }

        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .tabs {
          display: flex;
          background: var(--bg-tertiary);
          border-bottom: 1px solid var(--border);
        }

        .tab {
          flex: 1;
          padding: 1rem;
          font-family: 'Orbitron', sans-serif;
          font-size: 0.875rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 1px;
          border: none;
          background: transparent;
          color: var(--text-secondary);
          cursor: pointer;
          transition: all 0.3s ease;
          position: relative;
        }

        .tab.active {
          color: var(--accent-primary);
        }

        .tab.active::after {
          content: '';
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          height: 3px;
          background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        }

        .tab:hover {
          color: var(--accent-primary);
        }

        .tab-content {
          padding: 2rem;
        }

        .result-text {
          font-size: 1rem;
          line-height: 1.8;
          color: var(--text-primary);
          white-space: pre-wrap;
        }

        .error-message {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1rem;
          background: rgba(255, 51, 102, 0.1);
          border: 1px solid var(--accent-secondary);
          border-radius: 8px;
          color: var(--accent-secondary);
          margin-top: 1rem;
        }

        .metadata {
          display: flex;
          gap: 2rem;
          padding: 1rem;
          background: var(--bg-tertiary);
          border-radius: 8px;
          margin-bottom: 1rem;
          flex-wrap: wrap;
        }

        .metadata-item {
          display: flex;
          flex-direction: column;
        }

        .metadata-label {
          font-size: 0.75rem;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 1px;
          margin-bottom: 0.25rem;
        }

        .metadata-value {
          font-family: 'Orbitron', sans-serif;
          font-size: 0.875rem;
          color: var(--accent-primary);
          font-weight: 700;
        }

        .stat-badge {
          display: inline-block;
          padding: 0.25rem 0.75rem;
          background: rgba(0, 240, 255, 0.1);
          border: 1px solid var(--accent-primary);
          border-radius: 20px;
          font-size: 0.75rem;
          color: var(--accent-primary);
          font-weight: 700;
          margin: 0 0.5rem 0.5rem 0;
        }
      `}</style>

      <div className="content-wrapper">
        <header className="header">
          <h1 className="title">TACTICAL COMMAND</h1>
          <p className="subtitle">ESports Strategy Analyzer</p>
        </header>

        <div className="main-grid">
          <div className="panel">
            <div className="panel-header">
              <Shield />
              Match Data
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Team A</label>
                <input
                  type="text"
                  className="form-input"
                  value={matchData.team_a}
                  onChange={(e) => setMatchData({...matchData, team_a: e.target.value})}
                  placeholder="Enter team name"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Team B</label>
                <input
                  type="text"
                  className="form-input"
                  value={matchData.team_b}
                  onChange={(e) => setMatchData({...matchData, team_b: e.target.value})}
                  placeholder="Enter team name"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Winner</label>
                <input
                  type="text"
                  className="form-input"
                  value={matchData.winner}
                  onChange={(e) => setMatchData({...matchData, winner: e.target.value})}
                  placeholder="Winning team"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Duration</label>
                <input
                  type="text"
                  className="form-input"
                  value={matchData.duration}
                  onChange={(e) => setMatchData({...matchData, duration: e.target.value})}
                  placeholder="e.g., 42:15"
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Team A Composition</label>
              <input
                type="text"
                className="form-input"
                value={matchData.team_a_composition}
                onChange={(e) => setMatchData({...matchData, team_a_composition: e.target.value})}
                placeholder="Hero1, Hero2, Hero3, ..."
              />
            </div>

            <div className="form-group">
              <label className="form-label">Team B Composition</label>
              <input
                type="text"
                className="form-input"
                value={matchData.team_b_composition}
                onChange={(e) => setMatchData({...matchData, team_b_composition: e.target.value})}
                placeholder="Hero1, Hero2, Hero3, ..."
              />
            </div>

            <div className="form-group">
              <label className="form-label">Events (one per line)</label>
              <textarea
                className="form-textarea"
                value={matchData.events}
                onChange={(e) => setMatchData({...matchData, events: e.target.value})}
                placeholder="5:30 - First blood&#10;12:00 - Tower down&#10;..."
                rows={5}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Commentary</label>
              <textarea
                className="form-textarea"
                value={matchData.commentary}
                onChange={(e) => setMatchData({...matchData, commentary: e.target.value})}
                placeholder="Optional match commentary..."
                rows={3}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Statistics (Key: Value format)</label>
              <textarea
                className="form-textarea"
                value={matchData.statistics}
                onChange={(e) => setMatchData({...matchData, statistics: e.target.value})}
                placeholder="Team A Kills: 45&#10;Team B Kills: 32&#10;..."
                rows={4}
              />
            </div>

            <button className="btn btn-secondary" onClick={loadExample}>
              <Sparkles size={20} />
              Load Example Match
            </button>
          </div>

          <div className="panel">
            <div className="panel-header">
              <Target />
              Analysis Configuration
            </div>

            <div className="form-group">
              <label className="form-label">Analysis Type</label>
              <div className="analysis-type-grid">
                <div 
                  className={`analysis-type-card ${analysisType === 'complete' ? 'active' : ''}`}
                  onClick={() => setAnalysisType('complete')}
                >
                  <div className="analysis-type-icon"><Trophy size={24} /></div>
                  <div className="analysis-type-title">Complete</div>
                  <div className="analysis-type-desc">Full Analysis</div>
                </div>

                <div 
                  className={`analysis-type-card ${analysisType === 'summary' ? 'active' : ''}`}
                  onClick={() => setAnalysisType('summary')}
                >
                  <div className="analysis-type-icon"><Activity size={24} /></div>
                  <div className="analysis-type-title">Summary</div>
                  <div className="analysis-type-desc">Match Overview</div>
                </div>

                <div 
                  className={`analysis-type-card ${analysisType === 'recommendations' ? 'active' : ''}`}
                  onClick={() => setAnalysisType('recommendations')}
                >
                  <div className="analysis-type-icon"><TrendingUp size={24} /></div>
                  <div className="analysis-type-title">Tactics</div>
                  <div className="analysis-type-desc">Recommendations</div>
                </div>

                <div 
                  className={`analysis-type-card ${analysisType === 'turning-points' ? 'active' : ''}`}
                  onClick={() => setAnalysisType('turning-points')}
                >
                  <div className="analysis-type-icon"><Zap size={24} /></div>
                  <div className="analysis-type-title">Pivots</div>
                  <div className="analysis-type-desc">Key Moments</div>
                </div>
              </div>
            </div>

            {(analysisType === 'summary' || analysisType === 'complete') && (
              <>
                <div className="form-group">
                  <label className="form-label">Summary Style</label>
                  <select 
                    className="form-select"
                    value={style}
                    onChange={(e) => setStyle(e.target.value)}
                  >
                    <option value="analytical">Analytical</option>
                    <option value="narrative">Narrative</option>
                    <option value="bullet">Bullet Points</option>
                    <option value="tactical">Tactical</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Focus Area</label>
                  <select 
                    className="form-select"
                    value={focus}
                    onChange={(e) => setFocus(e.target.value)}
                  >
                    <option value="team_fights">Team Fights</option>
                    <option value="economy">Economy</option>
                    <option value="objectives">Objectives</option>
                    <option value="composition">Composition</option>
                    <option value="macro">Macro Strategy</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Summary Length</label>
                  <select 
                    className="form-select"
                    value={length}
                    onChange={(e) => setLength(e.target.value)}
                  >
                    <option value="short">Short</option>
                    <option value="medium">Medium</option>
                    <option value="long">Long</option>
                  </select>
                </div>
              </>
            )}

            {(analysisType === 'recommendations' || analysisType === 'complete') && (
              <div className="form-group">
                <label className="form-label">Team for Recommendations</label>
                <select 
                  className="form-select"
                  value={team}
                  onChange={(e) => setTeam(e.target.value)}
                >
                  <option value="team_a">Team A</option>
                  <option value="team_b">Team B</option>
                </select>
              </div>
            )}

            <div className="button-group">
              <button 
                className="btn btn-primary" 
                onClick={analyzeMatch}
                disabled={loading || !matchData.team_a || !matchData.team_b || !matchData.winner}
              >
                {loading ? (
                  <>
                    <div className="loading-spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play size={20} />
                    Analyze Match
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="error-message">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}
          </div>

          {results && analysisType === 'complete' && results.analysis && (
            <div className="results-panel">
              <div className="tabs">
                <button 
                  className={`tab ${activeTab === 'summary' ? 'active' : ''}`}
                  onClick={() => setActiveTab('summary')}
                >
                  <Activity size={16} /> Summary
                </button>
                <button 
                  className={`tab ${activeTab === 'recommendations' ? 'active' : ''}`}
                  onClick={() => setActiveTab('recommendations')}
                >
                  <TrendingUp size={16} /> Tactics
                </button>
                <button 
                  className={`tab ${activeTab === 'turning-points' ? 'active' : ''}`}
                  onClick={() => setActiveTab('turning-points')}
                >
                  <Zap size={16} /> Pivots
                </button>
              </div>

              <div className="tab-content">
                {results.metadata && results.metadata.match && (
                  <div className="metadata">
                    <div className="metadata-item">
                      <span className="metadata-label">Match</span>
                      <span className="metadata-value">
                        {results.metadata.match.team_a} vs {results.metadata.match.team_b}
                      </span>
                    </div>
                    <div className="metadata-item">
                      <span className="metadata-label">Winner</span>
                      <span className="metadata-value">{results.metadata.match.winner}</span>
                    </div>
                    <div className="metadata-item">
                      <span className="metadata-label">Analysis Time</span>
                      <span className="metadata-value">
                        {new Date(results.metadata.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                )}

                {activeTab === 'summary' && (
                  <div className="result-text">{results.analysis.summary}</div>
                )}

                {activeTab === 'recommendations' && (
                  <div className="result-text">{results.analysis.recommendations}</div>
                )}

                {activeTab === 'turning-points' && (
                  <div className="result-text">{results.analysis.turning_points}</div>
                )}
              </div>
            </div>
          )}

          {results && analysisType !== 'complete' && (
            <div className="results-panel">
              <div className="tab-content">
                {results.metadata && (
                  <div className="metadata">
                    {results.metadata.style && (
                      <div className="metadata-item">
                        <span className="metadata-label">Style</span>
                        <span className="metadata-value">{results.metadata.style}</span>
                      </div>
                    )}
                    {results.metadata.focus && (
                      <div className="metadata-item">
                        <span className="metadata-label">Focus</span>
                        <span className="metadata-value">{results.metadata.focus}</span>
                      </div>
                    )}
                    {results.metadata.team && (
                      <div className="metadata-item">
                        <span className="metadata-label">Team</span>
                        <span className="metadata-value">{results.metadata.team}</span>
                      </div>
                    )}
                    <div className="metadata-item">
                      <span className="metadata-label">Generated</span>
                      <span className="metadata-value">
                        {new Date(results.metadata.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                )}

                <div className="result-text">
                  {results.summary || results.recommendations || results.turning_points}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}