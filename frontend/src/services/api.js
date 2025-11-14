// frontend/src/services/api.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  healthCheck: async () => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  analyzeMatch: async (matchData, focusTeam = 'team_a', promptControls = null) => {
    try {
      const response = await api.post('/analyze', {
        match_data: matchData,
        focus_team: focusTeam,
        prompt_controls: promptControls
      });
      return response.data;
    } catch (error) {
      console.error('Analysis failed:', error);
      throw error;
    }
  },

  uploadMatchFile: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await api.post('/upload-match', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
  },

  getPromptOptions: async () => {
    try {
      const response = await api.get('/prompt-options');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch prompt options:', error);
      throw error;
    }
  }
};

export default api;