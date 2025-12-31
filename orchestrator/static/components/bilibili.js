/**
 * Bilibili chat management component
 */

import { apiCall } from '../utils/api.js';
import { formatTimestamp } from '../utils/helpers.js';
import { makeCollapsible } from '../utils/panel-collapse.js';

let messagesContainer = null;

export function initBilibiliPanel() {
  // Get DOM elements
  messagesContainer = document.getElementById('bilibili-messages');
  const bilibiliPanel = document.getElementById('bilibili-panel');

  // Make bilibili panel collapsible
  if (bilibiliPanel) {
    makeCollapsible(bilibiliPanel, 'bilibili', false);
  }
}

function renderBilibiliMessage(message, isSuperchat = false) {
  const messageDiv = document.createElement('div');
  messageDiv.className = isSuperchat
    ? 'bilibili-message superchat'
    : 'bilibili-message';
  messageDiv.dataset.id = message.id;

  const header = document.createElement('div');
  header.className = 'bilibili-message-header';

  const user = document.createElement('span');
  user.className = 'bilibili-user';
  user.textContent = message.user;

  const meta = document.createElement('div');
  meta.style.display = 'flex';
  meta.style.gap = '8px';
  meta.style.alignItems = 'center';

  if (isSuperchat && message.amount) {
    const amount = document.createElement('span');
    amount.className = 'bilibili-amount';
    amount.textContent = `¥${message.amount}`;
    meta.appendChild(amount);
  }

  const timestamp = document.createElement('span');
  timestamp.className = 'bilibili-timestamp';
  timestamp.textContent = formatTimestamp(message.timestamp);
  meta.appendChild(timestamp);

  header.appendChild(user);
  header.appendChild(meta);

  const content = document.createElement('div');
  content.className = 'bilibili-content';
  content.textContent = message.content;

  messageDiv.appendChild(header);
  messageDiv.appendChild(content);

  return messageDiv;
}

export function addBilibiliMessage(message, isSuperchat = false) {
  if (!messagesContainer) {
    console.warn('Bilibili messages container not initialized');
    return;
  }

  const emptyMsg = messagesContainer.querySelector('.empty-chat');
  if (emptyMsg) {
    emptyMsg.remove();
  }

  // Check if message already exists (prevent duplicates)
  const existing = messagesContainer.querySelector(`[data-id="${message.id}"]`);
  if (existing) {
    return;
  }

  // Add new message
  const messageEl = renderBilibiliMessage(message, isSuperchat);
  messagesContainer.appendChild(messageEl);

  // Auto-scroll to bottom
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // Limit message count (keep last 100 messages)
  const messages = messagesContainer.querySelectorAll('.bilibili-message');
  if (messages.length > 100) {
    messages[0].remove();
  }
}

export async function fetchBilibiliChat() {
  if (!messagesContainer) {
    console.warn('Bilibili messages container not initialized');
    return;
  }

  try {
    const data = await apiCall('/ui/bilibili/chat');

    if (!data.enabled) {
      // Show disabled state
      messagesContainer.innerHTML = '<div class="empty-chat">Bilibili chat not enabled</div>';
      return;
    }

    // Clear existing messages
    messagesContainer.innerHTML = '';

    if (data.danmaku.length === 0 && data.superchats.length === 0) {
      messagesContainer.innerHTML = '<div class="empty-chat">No messages yet</div>';
      return;
    }

    // Render all messages in chronological order
    const allMessages = [
      ...data.danmaku.map(m => ({ ...m, type: 'danmaku' })),
      ...data.superchats.map(m => ({ ...m, type: 'superchat' }))
    ].sort((a, b) => a.timestamp - b.timestamp);

    allMessages.forEach(msg => {
      const messageEl = renderBilibiliMessage(msg, msg.type === 'superchat');
      messagesContainer.appendChild(messageEl);
    });

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

  } catch (e) {
    console.error('Error fetching Bilibili chat:', e);
    messagesContainer.innerHTML = '<div class="empty-chat">Error loading chat</div>';
  }
}
