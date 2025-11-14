// src/components/MatchUpload.jsx
import { useState, useRef } from 'react';
import { Upload, FileJson, Sparkles } from 'lucide-react';

function MatchUpload({ onMatchUpload, onLoadSample }) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file.type !== "application/json") {
      alert("Please upload a JSON file");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        onMatchUpload(data);
      } catch (error) {
        alert("Invalid JSON file format");
        console.error(error);
      }
    };
    reader.readAsText(file);
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="upload-container">
      <div 
        className={`dropzone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleChange}
          style={{ display: 'none' }}
        />
        
        <FileJson size={48} className="upload-icon" />
        <p className="upload-text">
          <strong>Drop your match JSON file here</strong> or click to browse
        </p>
        <p className="upload-subtext">Only .json files are accepted</p>
      </div>

      <div className="upload-actions">
        <button 
          className="btn btn-primary"
          onClick={onButtonClick}
        >
          <Upload size={18} />
          Choose File
        </button>
        
        <button 
          className="btn btn-secondary"
          onClick={onLoadSample}
        >
          <Sparkles size={18} />
          Load Sample Data
        </button>
      </div>
    </div>
  );
}

export default MatchUpload;