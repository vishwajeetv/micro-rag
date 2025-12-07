import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Source {
  document_id: number;
  document_title: string;
  document_url: string;
  chunk_content: string;
  relevance_score: number;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  model: string;
  tokens_used: number;
  latency_ms: number;
}

export interface ChatRequest {
  question: string;
  collection_slug?: string;
  top_k?: number;
  include_sources?: boolean;
}

export interface StreamEvent {
  type: 'sources' | 'content' | 'done' | 'error';
  data: unknown;
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await api.post<ChatResponse>('/chat', {
    question: request.question,
    collection_slug: request.collection_slug,
    top_k: request.top_k ?? 5,
    include_sources: request.include_sources ?? true,
  });
  return response.data;
}

export async function* streamChatMessage(
  request: ChatRequest
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: request.question,
      collection_slug: request.collection_slug,
      top_k: request.top_k ?? 5,
      include_sources: request.include_sources ?? true,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to send message');
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        try {
          yield JSON.parse(data) as StreamEvent;
        } catch {
          // Skip invalid JSON
        }
      }
    }
  }
}

export default api;
