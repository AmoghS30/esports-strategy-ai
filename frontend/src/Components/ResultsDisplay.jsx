// frontend/src/components/ResultsDisplay.jsx
import { Download, Copy, Check, Settings } from 'lucide-react';
import { useState } from 'react';

function ResultsDisplay({ result, promptControls }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(result.result);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadAsText = () => {
    const element = document.createElement('a');
    const file = new Blob([result.result], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `${result.match_id}_comprehensive_analysis.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Parse and format markdown-like text
  const formatContent = (text) => {
    const lines = text.split('\n');
    const elements = [];
    let currentList = [];
    let listType = null;

    const flushList = () => {
      if (currentList.length > 0) {
        elements.push(
          <ul key={`list-${elements.length}`} className="formatted-list">
            {currentList.map((item, idx) => (
              <li key={idx} dangerouslySetInnerHTML={{ __html: item }} />
            ))}
          </ul>
        );
        currentList = [];
        listType = null;
      }
    };

    lines.forEach((line, index) => {
      const trimmedLine = line.trim();

      if (!trimmedLine) {
        flushList();
        return;
      }

      // Headers (###, ##, #)
      if (trimmedLine.startsWith('### ')) {
        flushList();
        elements.push(
          <h3 key={`h3-${index}`} className="formatted-h3">
            {trimmedLine.substring(4)}
          </h3>
        );
      } else if (trimmedLine.startsWith('## ')) {
        flushList();
        elements.push(
          <h2 key={`h2-${index}`} className="formatted-h2">
            {trimmedLine.substring(3)}
          </h2>
        );
      } else if (trimmedLine.startsWith('# ')) {
        flushList();
        elements.push(
          <h1 key={`h1-${index}`} className="formatted-h1">
            {trimmedLine.substring(2)}
          </h1>
        );
      }
      // Bullet points (-, *, •)
      else if (trimmedLine.match(/^[-*•]\s/)) {
        const content = trimmedLine.substring(2).trim();
        currentList.push(formatInlineStyles(content));
      }
      // Numbered lists (1., 2., etc.)
      else if (trimmedLine.match(/^\d+\.\s/)) {
        if (listType !== 'ordered') {
          flushList();
          listType = 'ordered';
        }
        const content = trimmedLine.replace(/^\d+\.\s/, '').trim();
        currentList.push(formatInlineStyles(content));
      }
      // Bold text surrounded by **
      else if (trimmedLine.includes('**')) {
        flushList();
        elements.push(
          <p key={`p-${index}`} className="formatted-paragraph" 
             dangerouslySetInnerHTML={{ __html: formatInlineStyles(trimmedLine) }} />
        );
      }
      // Regular paragraphs
      else {
        flushList();
        elements.push(
          <p key={`p-${index}`} className="formatted-paragraph" 
             dangerouslySetInnerHTML={{ __html: formatInlineStyles(trimmedLine) }} />
        );
      }
    });

    flushList();
    return elements;
  };

  // Format inline styles (bold, italic, code)
  const formatInlineStyles = (text) => {
    return text
      // Bold: **text** or __text__
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.+?)__/g, '<strong>$1</strong>')
      // Italic: *text* or _text_
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/_(.+?)_/g, '<em>$1</em>')
      // Inline code: `code`
      .replace(/`(.+?)`/g, '<code class="inline-code">$1</code>')
      // Links: [text](url)
      .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  };

  return (
    <div className="results-container">
      <div className="results-header">
        <h2>📊 Comprehensive Match Analysis</h2>
        <div className="results-actions">
          <button 
            className="btn btn-icon"
            onClick={copyToClipboard}
            title="Copy to clipboard"
          >
            {copied ? <Check size={18} /> : <Copy size={18} />}
          </button>
          <button 
            className="btn btn-icon"
            onClick={downloadAsText}
            title="Download as text"
          >
            <Download size={18} />
          </button>
        </div>
      </div>

      {promptControls && (
        <div className="controls-used">
          <Settings size={16} />
          <span>
            <strong>Analysis Settings:</strong> {promptControls.length} length • {promptControls.style} style • 
            {' '}{promptControls.focus.join(', ')} focus
          </span>
        </div>
      )}

      <div className="analysis-sections-info">
        <div className="section-badge">✅ Match Overview</div>
        <div className="section-badge">✅ Turning Points</div>
        <div className="section-badge">✅ Tactical Recommendations</div>
        <div className="section-badge">✅ Key Takeaways</div>
      </div>

      <div className="results-content formatted-content">
        {formatContent(result.result)}
      </div>

      <div className="results-meta">
        <span className="meta-item">Match ID: {result.match_id}</span>
        <span className="meta-item">Focus Team: {result.focus_team}</span>
      </div>
    </div>
  );
}

export default ResultsDisplay;