// src/components/AnalysisConfig.jsx
import { Play } from 'lucide-react';

function AnalysisConfig({ 
  analysisType, 
  setAnalysisType, 
  focusTeam, 
  setFocusTeam, 
  matchData,
  onAnalyze,
  loading,
  disabled 
}) {
  
  const analysisOptions = [
    { value: 'analytical', label: '📊 Analytical Summary', description: 'Data-driven analysis with key metrics' },
    { value: 'narrative', label: '📖 Narrative Summary', description: 'Story-like match recap' },
    { value: 'recommendations', label: '💡 Tactical Recommendations', description: 'Strategic improvements for selected team' },
    { value: 'turning_points', label: '🔄 Turning Points Analysis', description: 'Critical moments that decided the match' }
  ];

  return (
    <div className="analysis-config">
      <div className="config-section">
        <label className="config-label">Analysis Type</label>
        <div className="analysis-options">
          {analysisOptions.map((option) => (
            <div 
              key={option.value}
              className={`analysis-option ${analysisType === option.value ? 'selected' : ''}`}
              onClick={() => setAnalysisType(option.value)}
            >
              <div className="option-header">
                <span className="option-label">{option.label}</span>
                {analysisType === option.value && (
                  <div className="selected-indicator"></div>
                )}
              </div>
              <p className="option-description">{option.description}</p>
            </div>
          ))}
        </div>
      </div>

      {analysisType === 'recommendations' && matchData && (
        <div className="config-section">
          <label className="config-label">Focus Team</label>
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
      )}

      <button 
        className="btn btn-analyze"
        onClick={onAnalyze}
        disabled={disabled}
      >
        {loading ? (
          <>
            <div className="btn-spinner"></div>
            Analyzing...
          </>
        ) : (
          <>
            <Play size={18} />
            Analyze Match
          </>
        )}
      </button>
    </div>
  );
}

export default AnalysisConfig;