import { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Paper,
  Typography,
  CircularProgress,
  Chip,
  Collapse,
  Link,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ReactMarkdown from 'react-markdown';
import { streamChatMessage, type Source } from './services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  loading?: boolean;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    // Add placeholder for assistant response
    setMessages((prev) => [
      ...prev,
      { role: 'assistant', content: '', loading: true },
    ]);

    try {
      let sources: Source[] = [];
      let content = '';

      for await (const event of streamChatMessage({ question: userMessage, collection_slug: 'eu5-wiki' })) {
        if (event.type === 'sources') {
          sources = event.data as Source[];
        } else if (event.type === 'content') {
          content += event.data as string;
          setMessages((prev) => {
            const newMessages = [...prev];
            const lastMsg = newMessages[newMessages.length - 1];
            if (lastMsg.role === 'assistant') {
              lastMsg.content = content;
              lastMsg.sources = sources;
              lastMsg.loading = true;
            }
            return newMessages;
          });
        } else if (event.type === 'done') {
          setMessages((prev) => {
            const newMessages = [...prev];
            const lastMsg = newMessages[newMessages.length - 1];
            if (lastMsg.role === 'assistant') {
              lastMsg.loading = false;
            }
            return newMessages;
          });
        } else if (event.type === 'error') {
          throw new Error(event.data as string);
        }
      }
    } catch (error) {
      setMessages((prev) => {
        const newMessages = [...prev];
        const lastMsg = newMessages[newMessages.length - 1];
        if (lastMsg.role === 'assistant') {
          lastMsg.content = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
          lastMsg.loading = false;
        }
        return newMessages;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        maxWidth: 800,
        margin: '0 auto',
        p: 2,
      }}
    >
      <Typography variant="h5" sx={{ mb: 2, textAlign: 'center' }}>
        EU5 Wiki Chat
      </Typography>

      {/* Messages */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
        {messages.length === 0 && (
          <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
            Ask a question about Europa Universalis 5
          </Typography>
        )}

        {messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input */}
      <Paper
        component="form"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        sx={{ display: 'flex', alignItems: 'center', p: 1 }}
      >
        <TextField
          fullWidth
          placeholder="Ask about EU5..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          size="small"
          sx={{ mr: 1 }}
        />
        <IconButton type="submit" disabled={loading || !input.trim()}>
          {loading ? <CircularProgress size={24} /> : <SendIcon />}
        </IconButton>
      </Paper>
    </Box>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Paper
        sx={{
          p: 2,
          maxWidth: '80%',
          bgcolor: isUser ? 'primary.main' : 'grey.100',
          color: isUser ? 'white' : 'text.primary',
        }}
      >
        {isUser ? (
          <Typography>{message.content}</Typography>
        ) : (
          <Box sx={{
            '& p': { m: 0, mb: 1 },
            '& ul, & ol': { m: 0, pl: 2 },
            '& li': { mb: 0.5 },
            '& strong': { fontWeight: 600 },
          }}>
            <ReactMarkdown>{message.content}</ReactMarkdown>
            {message.loading && <CircularProgress size={16} sx={{ ml: 1 }} />}
          </Box>
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <Box sx={{ mt: 1 }}>
            <Chip
              size="small"
              label={`${message.sources.length} sources`}
              onClick={() => setShowSources(!showSources)}
              icon={showSources ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              sx={{ cursor: 'pointer' }}
            />
            <Collapse in={showSources}>
              <Box sx={{ mt: 1 }}>
                {message.sources.map((source, idx) => (
                  <Box key={idx} sx={{ mb: 1, fontSize: '0.85rem' }}>
                    <Link
                      href={source.document_url}
                      target="_blank"
                      rel="noopener"
                    >
                      {source.document_title}
                    </Link>
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                      ({(source.relevance_score * 100).toFixed(0)}% match)
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Collapse>
          </Box>
        )}
      </Paper>
    </Box>
  );
}
