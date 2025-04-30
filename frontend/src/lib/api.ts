import axios from 'axios';

interface SentimentResult {
  sentiment: string;
  probabilities: Record<string, number>;
  confidence: number;
}

interface HealthCheckResponse {
  status: string;
}

const api = axios.create({
  baseURL: '/api'
});

export const sentimentService = {
  analyze: async (text: string): Promise<SentimentResult> => {
    try {
      const response = await api.post<SentimentResult>('/analyze', { text });
      return response.data;
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
      throw error;
    }
  },
  
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