/**
 * History management component
 */

import { elements } from '../utils/dom.js';
import { formatTimestamp } from '../utils/helpers.js';
import { apiCall } from '../utils/api.js';

function renderHistoryMessage(message) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${message.role}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = message.content;
  
  const timestamp = document.createElement('div');
  timestamp.className = 'message-timestamp';
  timestamp.textContent = formatTimestamp(message.timestamp);
  
  messageDiv.appendChild(bubble);
  messageDiv.appendChild(timestamp);
  
  return messageDiv;
}

export function renderHistory(history) {
  const emptyMsg = elements.historyPanel.querySelector('.empty-history');
  if (emptyMsg) {
    emptyMsg.remove();
  }
  
  elements.historyPanel.innerHTML = '';
  
  if (!history || history.length === 0) {
    elements.historyPanel.innerHTML = '<div class="empty-history">No conversation history yet</div>';
    return;
  }
  
  history.forEach(msg => {
    elements.historyPanel.appendChild(renderHistoryMessage(msg));
  });
  
  elements.historyPanel.scrollTop = elements.historyPanel.scrollHeight;
}

export async function fetchHistory() {
  try {
    const data = await apiCall('/ui/history');
    if (data.history) {
      renderHistory(data.history);
    }
  } catch (e) {
    console.error('Error fetching history:', e);
  }
}

