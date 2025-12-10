import { useState } from 'react';
import { ThemeProvider, createTheme, CssBaseline, Tabs, Tab, Box } from '@mui/material';
import Chat from './Chat';
import AgentChat from './AgentChat';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
  },
});

function App() {
  const [tab, setTab] = useState(1); // Default to Agent Chat

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper' }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)} centered>
          <Tab label="RAG Chat" />
          <Tab label="Agent Chat" />
        </Tabs>
      </Box>
      {tab === 0 && <Chat />}
      {tab === 1 && <AgentChat />}
    </ThemeProvider>
  );
}

export default App;
