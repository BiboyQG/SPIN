import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { startExtraction, getExtractionStatus } from '../api/client';
import { ExtractionResponse } from '../types/api';
import { getStoredSettings, POLLING_INTERVAL } from '../lib/utils';
import { JsonView } from 'react-json-view-lite';
import 'react-json-view-lite/dist/index.css';

export function Home() {
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [depth, setDepth] = useState(1);
  const [isExtracting, setIsExtracting] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [extractionStatus, setExtractionStatus] = useState<ExtractionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const settings = getStoredSettings();
    if (!settings.openai_base_url || !settings.model_name) {
      navigate('/settings');
    }
  }, [navigate]);

  useEffect(() => {
    let pollInterval: ReturnType<typeof setInterval>;

    const pollStatus = async () => {
      if (!taskId) return;

      try {
        const status = await getExtractionStatus(taskId);
        setExtractionStatus(status);

        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(pollInterval);
          setIsExtracting(false);
          if (status.status === 'failed') {
            setError(status.error || 'Extraction failed');
          }
        }
      } catch {
        clearInterval(pollInterval);
        setIsExtracting(false);
        setError('Failed to get extraction status');
      }
    };

    if (isExtracting && taskId) {
      pollInterval = setInterval(pollStatus, POLLING_INTERVAL);
    }

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [taskId, isExtracting]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsExtracting(true);
    setExtractionStatus(null);

    try {
      const settings = getStoredSettings();
      const response = await startExtraction({
        input,
        depth,
        openai_base_url: settings.openai_base_url,
        model_name: settings.model_name,
      });
      setTaskId(response.task_id);
    } catch {
      setIsExtracting(false);
      setError('Failed to start extraction');
    }
  };

  const parseJsonResult = (result: Record<string, unknown> | string): Record<string, unknown> => {
    if (typeof result === 'string') {
      try {
        return JSON.parse(result);
      } catch {
        return { value: result };
      }
    }
    
    // Handle nested JSON strings in object values
    const parsedResult = { ...result };
    Object.entries(result).forEach(([key, value]) => {
      if (typeof value === 'string') {
        try {
          parsedResult[key] = JSON.parse(value);
        } catch {
          // Keep original string value if it's not valid JSON
          parsedResult[key] = value;
        }
      }
    });
    
    return parsedResult;
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Information Extraction</h1>

        <form onSubmit={handleSubmit} className="space-y-6 mb-8">
          <div>
            <label htmlFor="input" className="block text-sm font-medium mb-1">
              Search Query or URLs
            </label>
            <textarea
              id="input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="w-full px-3 py-2 border rounded-md h-32"
              required
            />
          </div>

          <div>
            <label htmlFor="depth" className="block text-sm font-medium mb-1">
              Search Depth
            </label>
            <input
              type="number"
              id="depth"
              value={depth}
              onChange={(e) => setDepth(parseInt(e.target.value))}
              min={0}
              className="w-full px-3 py-2 border rounded-md"
              required
            />
          </div>

          <Button
            type="submit"
            disabled={isExtracting}
            className="w-full"
          >
            {isExtracting ? 'Extracting...' : 'Start Extraction'}
          </Button>
        </form>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}

        {extractionStatus?.progress && (
          <div className="mb-6">
            <div className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-3 rounded">
              Processing URL {extractionStatus.progress.url_number} of {extractionStatus.progress.total_urls}:
              {' '}{extractionStatus.progress.current_url}
            </div>
          </div>
        )}

        {extractionStatus?.result && (
          <div className="relative bg-white p-4 rounded-lg shadow-sm border">
            <button
              onClick={() => navigator.clipboard.writeText(JSON.stringify(extractionStatus.result, null, 2))}
              className="absolute top-2 right-2 bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded text-sm"
            >
              Copy
            </button>
            <div className="font-mono text-sm overflow-auto mt-4">
              <JsonView data={parseJsonResult(extractionStatus.result)} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 