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
  LinearProgress,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import SearchIcon from '@mui/icons-material/Search';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import ReactMarkdown from 'react-markdown';
import { streamAgentMessage, type AgentEvent } from './services/api';

// Tool execution step shown in the UI
interface ToolStep {
  tool: string;
  arguments: string;
  status: 'running' | 'success' | 'error';
  output?: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  toolSteps?: ToolStep[];
  loading?: boolean;
  thinking?: boolean;
  iteration?: number;
  tokensSoFar?: number;
  tokensUsed?: number;
}

export default function AgentChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
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
      { role: 'assistant', content: '', loading: true, toolSteps: [] },
    ]);

    try {
      let content = '';
      let toolSteps: ToolStep[] = [];

      for await (const event of streamAgentMessage({
        message: userMessage,
        session_id: sessionId,
        collection_slug: 'eu5-wiki'
      })) {
        handleAgentEvent(event, content, toolSteps, (newContent, newSteps, thinking, iteration, tokensSoFar) => {
          content = newContent;
          toolSteps = newSteps;
          setMessages((prev) => {
            const newMessages = [...prev];
            const lastMsg = newMessages[newMessages.length - 1];
            if (lastMsg.role === 'assistant') {
              lastMsg.content = content;
              lastMsg.toolSteps = [...toolSteps];
              lastMsg.thinking = thinking;
              lastMsg.iteration = iteration;
              lastMsg.tokensSoFar = tokensSoFar;
              lastMsg.loading = true;
            }
            return [...newMessages];
          });
        });

        // Handle metadata (session_id)
        if (event.type === 'metadata' && event.data.session_id) {
          setSessionId(event.data.session_id);
        }

        // Handle final answer
        if (event.type === 'answer') {
          content = event.data.content || '';
          const tokensUsed = event.data.tokens_used;
          setMessages((prev) => {
            const newMessages = [...prev];
            const lastMsg = newMessages[newMessages.length - 1];
            if (lastMsg.role === 'assistant') {
              lastMsg.content = content;
              lastMsg.tokensUsed = tokensUsed;
              lastMsg.loading = false;
              lastMsg.thinking = false;
            }
            return [...newMessages];
          });
        }

        // Handle error
        if (event.type === 'error') {
          throw new Error(event.data.message || 'Unknown error');
        }
      }
    } catch (error) {
      setMessages((prev) => {
        const newMessages = [...prev];
        const lastMsg = newMessages[newMessages.length - 1];
        if (lastMsg.role === 'assistant') {
          lastMsg.content = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
          lastMsg.loading = false;
          lastMsg.thinking = false;
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
        AI Agent Chat
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ mb: 2, textAlign: 'center', display: 'block' }}>
        Powered by ReAct pattern - Watch the agent think and use tools
      </Typography>

      {/* Messages */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
        {messages.length === 0 && (
          <Typography color="text.secondary" sx={{ textAlign: 'center', mt: 4 }}>
            Ask a question - the agent will search the knowledge base and reason about the answer
          </Typography>
        )}

        {messages.map((msg, idx) => (
          <AgentMessageBubble key={idx} message={msg} />
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
          placeholder="Ask the agent..."
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

function handleAgentEvent(
  event: AgentEvent,
  currentContent: string,
  currentSteps: ToolStep[],
  onUpdate: (content: string, steps: ToolStep[], thinking: boolean, iteration?: number, tokensSoFar?: number) => void
) {
  switch (event.type) {
    case 'thinking':
      onUpdate(currentContent, currentSteps, true, event.data.iteration, event.data.tokens_so_far);
      break;

    case 'tool_call':
      // Add new tool step
      const newStep: ToolStep = {
        tool: event.data.tool || 'unknown',
        arguments: event.data.arguments || '{}',
        status: 'running',
      };
      onUpdate(currentContent, [...currentSteps, newStep], false);
      break;

    case 'tool_result':
      // Update last tool step with result
      if (currentSteps.length > 0) {
        const updatedSteps = [...currentSteps];
        const lastStep = updatedSteps[updatedSteps.length - 1];
        lastStep.status = event.data.success ? 'success' : 'error';
        lastStep.output = event.data.output;
        onUpdate(currentContent, updatedSteps, false);
      }
      break;
  }
}

function AgentMessageBubble({ message }: { message: Message }) {
  const [showTools, setShowTools] = useState(true);
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
          maxWidth: '85%',
          bgcolor: isUser ? 'primary.main' : 'grey.100',
          color: isUser ? 'white' : 'text.primary',
        }}
      >
        {isUser ? (
          <Typography>{message.content}</Typography>
        ) : (
          <Box>
            {/* Thinking indicator */}
            {message.thinking && (
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Thinking... (iteration {message.iteration})
                    {message.tokensSoFar ? ` · ${message.tokensSoFar.toLocaleString()} tokens` : ''}
                  </Typography>
                </Box>
                <LinearProgress />
              </Box>
            )}

            {/* Tool Steps */}
            {message.toolSteps && message.toolSteps.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Chip
                  size="small"
                  icon={<SearchIcon />}
                  label={`${message.toolSteps.length} tool${message.toolSteps.length > 1 ? 's' : ''} used`}
                  onClick={() => setShowTools(!showTools)}
                  deleteIcon={showTools ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  onDelete={() => setShowTools(!showTools)}
                  sx={{ cursor: 'pointer', mb: 1 }}
                />
                <Collapse in={showTools}>
                  <Box sx={{ pl: 1, borderLeft: '2px solid #ddd' }}>
                    {message.toolSteps.map((step, idx) => (
                      <ToolStepDisplay key={idx} step={step} />
                    ))}
                  </Box>
                </Collapse>
              </Box>
            )}

            {/* Answer content */}
            {message.content && (
              <Box sx={{
                '& p': { m: 0, mb: 1 },
                '& ul, & ol': { m: 0, pl: 2 },
                '& li': { mb: 0.5 },
                '& strong': { fontWeight: 600 },
              }}>
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </Box>
            )}

            {/* Token usage display */}
            {!message.loading && message.tokensUsed !== undefined && message.tokensUsed > 0 && (
              <Box sx={{ mt: 1.5, pt: 1, borderTop: '1px solid #e0e0e0' }}>
                <Typography variant="caption" color="text.secondary">
                  {message.tokensUsed.toLocaleString()} tokens used
                </Typography>
              </Box>
            )}

            {/* Loading indicator for content */}
            {message.loading && !message.thinking && !message.content && (
              <CircularProgress size={16} />
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
}

function ToolStepDisplay({ step }: { step: ToolStep }) {
  const [expanded, setExpanded] = useState(false);

  // Parse arguments for display
  let parsedArgs: Record<string, unknown> = {};
  try {
    parsedArgs = JSON.parse(step.arguments);
  } catch {
    // ignore
  }

  return (
    <Box sx={{ mb: 1.5, fontSize: '0.85rem' }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          cursor: 'pointer',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        {step.status === 'running' && <CircularProgress size={14} />}
        {step.status === 'success' && <CheckCircleIcon sx={{ fontSize: 16, color: 'success.main' }} />}
        {step.status === 'error' && <ErrorIcon sx={{ fontSize: 16, color: 'error.main' }} />}

        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          {step.tool}
        </Typography>

        {parsedArgs.query && (
          <Typography variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
            "{String(parsedArgs.query).slice(0, 30)}..."
          </Typography>
        )}

        {expanded ? <ExpandLessIcon sx={{ fontSize: 16 }} /> : <ExpandMoreIcon sx={{ fontSize: 16 }} />}
      </Box>

      <Collapse in={expanded}>
        <Box sx={{ mt: 0.5, pl: 2, bgcolor: 'grey.50', p: 1, borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Arguments: {step.arguments}
          </Typography>
          {step.output && (
            <Typography
              variant="caption"
              component="pre"
              sx={{
                mt: 0.5,
                whiteSpace: 'pre-wrap',
                maxHeight: 150,
                overflow: 'auto',
                display: 'block',
              }}
            >
              {step.output}
            </Typography>
          )}
        </Box>
      </Collapse>
    </Box>
  );
}
