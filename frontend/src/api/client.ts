import axios from 'axios';
import { ExtractionRequest, ExtractionResponse } from '../types/api';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const startExtraction = async (request: ExtractionRequest): Promise<ExtractionResponse> => {
  const response = await api.post<ExtractionResponse>('/extract', request);
  return response.data;
};

export const getExtractionStatus = async (taskId: string): Promise<ExtractionResponse> => {
  const response = await api.get<ExtractionResponse>(`/status/${taskId}`);
  return response.data;
};

export const getSchemas = async (): Promise<string[]> => {
  const response = await api.get<{ schemas: string[] }>('/schemas');
  return response.data.schemas;
};

export const createSchema = async (name: string, content: string): Promise<{ message: string }> => {
  const response = await api.post<{ message: string }>(`/schemas/${name}`, content);
  return response.data;
}; 