const API_BASE = 'http://127.0.0.1:8000/api';

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

export interface StreamEvent {
  type: 'sources' | 'content' | 'done' | 'error';
  data: unknown;
}

export async function sendChatMessage(
  question: string,
  collectionSlug?: string
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      collection_slug: collectionSlug,
      top_k: 5,
      include_sources: true,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to send message');
  }

  return response.json();
}

export async function* streamChatMessage(
  question: string,
  collectionSlug?: string
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      collection_slug: collectionSlug,
      top_k: 5,
      include_sources: true,
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
