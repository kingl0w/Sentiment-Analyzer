import axios from 'axios';

// Define types for our API responses
interface SentimentResult {
  sentiment: string;
  probabilities: Record<string, number>;
  confidence: number;
}

interface HealthCheckResponse {
  status: string;
}

// Create axios instance with base URL
const api = axios.create({
  baseURL: 'http://backend:8000'
});

// Sentiment analysis service
export const sentimentService = {
  // Analyze sentiment of text
  analyze: async (text: string): Promise<SentimentResult> => {
    try {
      const response = await api.post<SentimentResult>('/analyze', { text });
      return response.data;
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
      throw error;
    }
  },
  
  // Check if API is available
  checkHealth: async (): Promise<boolean> => {
    try {
      const response = await api.get<HealthCheckResponse>('/health');
      return response.data.status === 'ok';
    } catch (error) {
      console.error('API health check failed:', error);
      return false;
    }
  }
};