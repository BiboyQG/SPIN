export interface ExtractionRequest {
  input: string;
  depth: number;
  openai_base_url: string;
  model_name: string;
  schema_type?: string;
}

export interface ExtractionProgress {
  stage: 'initializing' | 'scraping' | 'schema_detection' | 'schema_generation' | 'initial_extraction' | 'analyzing_fields' | 'gathering_links' | 'updating_data' | 'finalizing' | 'completed' | 'failed';
  stage_progress: number;
  current_url: string | null;
  url_number: number;
  total_urls: number;
  message: string;
  schema_type?: string;
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
