import React, { useState, useEffect } from 'react';
import { Button } from '../components/ui/button';
import { getStoredSettings, saveSettings } from '../lib/utils';
import { Settings as SettingsType } from '../types/api';

export function Settings() {
  const [settings, setSettings] = useState<SettingsType>(() => getStoredSettings());
  const [isSaving, setIsSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    try {
      saveSettings(settings);
      // Show success toast here
    } catch (error) {
      // Show error toast here
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Settings</h1>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="openai_base_url" className="block text-sm font-medium mb-1">
                OpenAI Base URL
              </label>
              <input
                type="url"
                id="openai_base_url"
                value={settings.openai_base_url}
                onChange={(e) => setSettings(prev => ({ ...prev, openai_base_url: e.target.value }))}
                className="w-full px-3 py-2 border rounded-md"
                required
              />
            </div>

            <div>
              <label htmlFor="model_name" className="block text-sm font-medium mb-1">
                Model Name
              </label>
              <input
                type="text"
                id="model_name"
                value={settings.model_name}
                onChange={(e) => setSettings(prev => ({ ...prev, model_name: e.target.value }))}
                className="w-full px-3 py-2 border rounded-md"
                required
              />
            </div>
          </div>

          <Button
            type="submit"
            disabled={isSaving}
            className="w-full"
          >
            {isSaving ? 'Saving...' : 'Save Settings'}
          </Button>
        </form>
      </div>
    </div>
  );
} 