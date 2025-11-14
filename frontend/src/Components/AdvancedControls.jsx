// frontend/src/components/AdvancedControls.jsx
import { useState } from 'react';
import { Settings, ChevronDown, ChevronUp } from 'lucide-react';

function AdvancedControls({ controls, setControls }) {
  const [expanded, setExpanded] = useState(false);

  const lengthOptions = [
    { value: 'short', label: 'Short', description: 'Concise (~800 words)' },
    { value: 'medium', label: 'Medium', description: 'Balanced (~1200 words)' },
    { value: 'long', label: 'Long', description: 'Detailed (~2000 words)' },
    { value: 'detailed', label: 'Detailed', description: 'In-depth (~3000+ words)' }
  ];

  const focusOptions = [
    { value: 'overall', label: 'Overall Analysis', icon: '🎯' },
    { value: 'team_fights', label: 'Team Fights', icon: '⚔️' },
    { value: 'economy', label: 'Economy', icon: '💰' },
    { value: 'objectives', label: 'Objectives', icon: '🎖️' },
    { value: 'positioning', label: 'Positioning', icon: '🗺️' },
    { value: 'draft', label: 'Draft/Picks', icon: '📋' }
  ];

  const styleOptions = [
    { value: 'analytical', label: 'Analytical', description: 'Data-driven, factual' },
    { value: 'narrative', label: 'Narrative', description: 'Story-telling, engaging' },
    { value: 'technical', label: 'Technical', description: 'Deep, expert terminology' },
    { value: 'casual', label: 'Casual', description: 'Easy to understand' }
  ];

  const toggleFocus = (focus) => {
    const currentFocus = controls.focus || ['overall'];
    let newFocus;
    
    if (focus === 'overall') {
      newFocus = ['overall'];
    } else {
      const filtered = currentFocus.filter(f => f !== 'overall');
      if (filtered.includes(focus)) {
        newFocus = filtered.filter(f => f !== focus);
        if (newFocus.length === 0) newFocus = ['overall'];
      } else {
        newFocus = [...filtered, focus];
      }
    }
    
    setControls({ ...controls, focus: newFocus });
  };

  return (
    <div className="advanced-controls">
      <button 
        className="advanced-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        <Settings size={18} />
        <span>Advanced Controls</span>
        {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </button>

      {expanded && (
        <div className="advanced-content">
          
          <div className="control-section">
            <label className="control-label">📏 Output Length</label>
            <div className="option-grid">
              {lengthOptions.map(option => (
                <div
                  key={option.value}
                  className={`option-card ${controls.length === option.value ? 'selected' : ''}`}
                  onClick={() => setControls({ ...controls, length: option.value })}
                >
                  <div className="option-title">{option.label}</div>
                  <div className="option-desc">{option.description}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="control-section">
            <label className="control-label">🎯 Focus Areas (select multiple)</label>
            <div className="focus-grid">
              {focusOptions.map(option => (
                <button
                  key={option.value}
                  className={`focus-chip ${
                    (controls.focus || ['overall']).includes(option.value) ? 'selected' : ''
                  }`}
                  onClick={() => toggleFocus(option.value)}
                >
                  <span className="focus-icon">{option.icon}</span>
                  <span>{option.label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="control-section">
            <label className="control-label">✍️ Writing Style</label>
            <div className="option-grid">
              {styleOptions.map(option => (
                <div
                  key={option.value}
                  className={`option-card ${controls.style === option.value ? 'selected' : ''}`}
                  onClick={() => setControls({ ...controls, style: option.value })}
                >
                  <div className="option-title">{option.label}</div>
                  <div className="option-desc">{option.description}</div>
                </div>
              ))}
            </div>
          </div>

          <button 
            className="btn btn-secondary reset-btn"
            onClick={() => setControls({
              length: 'medium',
              focus: ['overall'],
              style: 'analytical'
            })}
          >
            Reset to Defaults
          </button>
        </div>
      )}
    </div>
  );
}

export default AdvancedControls;