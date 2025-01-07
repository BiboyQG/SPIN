import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export const POLLING_INTERVAL = 1000; // 1 second

export function formatJson(obj: any): string {
  return JSON.stringify(obj, null, 2);
}

export function getStoredSettings() {
  const settings = localStorage.getItem('extraction-settings');
  return settings ? JSON.parse(settings) : {
    openai_base_url: '',
    model_name: ''
  };
}

export function saveSettings(settings: { openai_base_url: string; model_name: string }) {
  localStorage.setItem('extraction-settings', JSON.stringify(settings));
} 