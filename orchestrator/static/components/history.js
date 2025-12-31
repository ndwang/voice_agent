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

  // Handle different message types
  if (message.role === 'tool') {
    // Tool result message
    bubble.className += ' tool-result';
    const toolName = document.createElement('div');
    toolName.className = 'tool-name';
    toolName.textContent = `🔧 ${message.name}`;
    bubble.appendChild(toolName);

    const toolContent = document.createElement('div');
    toolContent.className = 'tool-content';
    toolContent.textContent = message.content;
    bubble.appendChild(toolContent);
  } else if (message.tool_calls && message.tool_calls.length > 0) {
    // Assistant message with tool calls
    bubble.className += ' tool-calls';
    message.tool_calls.forEach(tc => {
      const toolCall = document.createElement('div');
      toolCall.className = 'tool-call';
      toolCall.textContent = `🔧 Calling ${tc.name}(${JSON.stringify(tc.arguments)})`;
      bubble.appendChild(toolCall);
    });
  } else {
    // Regular message with text content
    bubble.textContent = message.content || '';
  }

  const timestamp = document.createElement('div');
  timestamp.className = 'message-timestamp';
  timestamp.textContent = formatTimestamp(message.timestamp);

  messageDiv.appendChild(bubble);
  messageDiv.appendChild(timestamp);

  return messageDiv;
}

let streamingMessages = {
  user: null,
  assistant: null
};

export function clearStreamingMessages() {
  streamingMessages.user = null;
  streamingMessages.assistant = null;
}

function smoothTextTransition(element, fromText, toText, duration = 400) {
  return new Promise((resolve) => {
    if (fromText === toText) {
      resolve();
      return;
    }

    // Add transition class for smooth morphing
    element.classList.add('text-transitioning');
    
    // Use requestAnimationFrame for smooth animation
    const startTime = performance.now();
    
    function animate(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function for smooth transition
      const ease = progress < 0.5 
        ? 2 * progress * progress 
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;
      
      // Interpolate between texts (simple approach: fade between)
      if (progress < 0.5) {
        // First half: fade out old text slightly
        element.style.opacity = String(1 - ease * 0.3);
      } else {
        // Second half: update text and fade in
        if (element.textContent !== toText) {
          element.textContent = toText;
        }
        element.style.opacity = String(0.7 + (ease - 0.5) * 0.3);
      }
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        // Ensure final state
        element.textContent = toText;
        element.style.opacity = '1';
        element.classList.remove('text-transitioning');
        resolve();
      }
    }
    
    requestAnimationFrame(animate);
  });
}

export function updateStreamingMessage(role, content) {
  const emptyMsg = elements.historyPanel.querySelector('.empty-history');
  if (emptyMsg) {
    emptyMsg.remove();
  }

  if (!streamingMessages[role]) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role} streaming`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    messageDiv.appendChild(bubble);
    elements.historyPanel.appendChild(messageDiv);
    streamingMessages[role] = messageDiv;
  }

  const bubble = streamingMessages[role].querySelector('.message-bubble');
  // Only update if not currently transitioning
  if (!bubble.classList.contains('text-transitioning')) {
    bubble.textContent = content;
  }
  
  elements.historyPanel.scrollTop = elements.historyPanel.scrollHeight;
}

export async function renderHistory(history) {
  const emptyMsg = elements.historyPanel.querySelector('.empty-history');
  if (emptyMsg) {
    emptyMsg.remove();
  }
  
  // Check if we have a streaming assistant message that needs to be transitioned
  const streamingAssistant = streamingMessages.assistant;
  let transitionedElement = null;
  let lastAssistantIndex = -1;
  
  if (streamingAssistant && history && history.length > 0) {
    // Find the last assistant message in history (should be the one we're streaming)
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].role === 'assistant') {
        lastAssistantIndex = i;
        break;
      }
    }
    
    if (lastAssistantIndex >= 0) {
      const lastAssistantMsg = history[lastAssistantIndex];
      const bubble = streamingAssistant.querySelector('.message-bubble');
      const currentText = bubble.textContent;
      const finalText = lastAssistantMsg.content;
      
      // If texts differ, smoothly transition
      if (currentText !== finalText) {
        await smoothTextTransition(bubble, currentText, finalText);
      }
      
      // Convert streaming message to proper history message
      streamingAssistant.classList.remove('streaming');
      // Remove any existing timestamp
      const existingTimestamp = streamingAssistant.querySelector('.message-timestamp');
      if (existingTimestamp) {
        existingTimestamp.remove();
      }
      const timestamp = document.createElement('div');
      timestamp.className = 'message-timestamp';
      timestamp.textContent = formatTimestamp(lastAssistantMsg.timestamp);
      streamingAssistant.appendChild(timestamp);
      transitionedElement = streamingAssistant;
    }
  }
  
  // Also handle user message if streaming
  const streamingUser = streamingMessages.user;
  let transitionedUserIndex = -1;
  let transitionedUserElement = null;
  
  if (streamingUser && history && history.length > 0) {
    // Find the last user message in history
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].role === 'user') {
        transitionedUserIndex = i;
        break;
      }
    }
    
    if (transitionedUserIndex >= 0) {
      const lastUserMsg = history[transitionedUserIndex];
      const bubble = streamingUser.querySelector('.message-bubble');
      // User messages typically don't change, but handle it just in case
      const currentText = bubble.textContent.replace(/\n\[interim\].*$/, '');
      const finalText = lastUserMsg.content;
      
      if (currentText !== finalText) {
        await smoothTextTransition(bubble, currentText, finalText);
      }
      
      streamingUser.classList.remove('streaming');
      const existingTimestamp = streamingUser.querySelector('.message-timestamp');
      if (existingTimestamp) {
        existingTimestamp.remove();
      }
      const timestamp = document.createElement('div');
      timestamp.className = 'message-timestamp';
      timestamp.textContent = formatTimestamp(lastUserMsg.timestamp);
      streamingUser.appendChild(timestamp);
      transitionedUserElement = streamingUser;
    }
  }
  
  // Clear streaming message references (they're now part of history)
  clearStreamingMessages();
  
  // Now rebuild history, preserving transitioned elements
  // Store transitioned elements temporarily
  const tempElements = [];
  if (transitionedElement) {
    tempElements.push(transitionedElement);
    transitionedElement.remove();
  }
  if (transitionedUserElement && transitionedUserElement !== transitionedElement) {
    tempElements.push(transitionedUserElement);
    transitionedUserElement.remove();
  }
  
  // Clear and render all messages
  elements.historyPanel.innerHTML = '';
  
  if (!history || history.length === 0) {
    elements.historyPanel.innerHTML = '<div class="empty-history">No conversation history yet</div>';
    return;
  }
  
  // Render all messages, using transitioned elements where appropriate
  history.forEach((msg, index) => {
    if (index === lastAssistantIndex && transitionedElement) {
      // Use the transitioned element instead of creating a new one
      elements.historyPanel.appendChild(transitionedElement);
    } else if (index === transitionedUserIndex && transitionedUserElement) {
      // Use the transitioned user element
      elements.historyPanel.appendChild(transitionedUserElement);
    } else {
      // Render normally
      elements.historyPanel.appendChild(renderHistoryMessage(msg));
    }
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

