// src/components/MatchPreview.jsx
import { CheckCircle, Users, Swords } from 'lucide-react';

function MatchPreview({ matchData }) {
  return (
    <div className="match-preview">
      <div className="preview-header">
        <CheckCircle size={20} className="check-icon" />
        <h3>Match Loaded Successfully</h3>
      </div>
      
      <div className="preview-content">
        <div className="preview-item">
          <span className="preview-label">Match ID:</span>
          <span className="preview-value">{matchData.match_id}</span>
        </div>
        
        <div className="preview-teams">
          <div className="team team-a">
            <Users size={18} />
            <span>{matchData.teams.team_a}</span>
          </div>
          
          <Swords size={18} className="vs-icon" />
          
          <div className="team team-b">
            <Users size={18} />
            <span>{matchData.teams.team_b}</span>
          </div>
        </div>

        <div className="preview-picks">
          <div className="picks-section">
            <h4>{matchData.teams.team_a} Picks:</h4>
            <div className="hero-list">
              {matchData.hero_picks.team_a.map((hero, idx) => (
                <span key={idx} className="hero-badge">{hero}</span>
              ))}
            </div>
          </div>
          
          <div className="picks-section">
            <h4>{matchData.teams.team_b} Picks:</h4>
            <div className="hero-list">
              {matchData.hero_picks.team_b.map((hero, idx) => (
                <span key={idx} className="hero-badge">{hero}</span>
              ))}
            </div>
          </div>
        </div>

        <div className="preview-events">
          <h4>Key Events: {matchData.events.length}</h4>
        </div>
      </div>
    </div>
  );
}

export default MatchPreview;