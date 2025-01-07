export interface ExtractionRequest {
  input: string;
  depth: number;
  openai_base_url: string;
  model_name: string;
}

export interface ExtractionProgress {
  current_url: string;
  url_number: number;
  total_urls: number;
}

export interface ExtractionResponse {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: ExtractionProgress;
  result?: Record<string, any>;
  error?: string;
}

export interface Settings {
  openai_base_url: string;
  model_name: string;
} 