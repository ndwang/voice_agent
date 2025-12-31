/**
 * Tools Panel Component
 * Displays all available tools with toggle switches
 */

import { apiCall } from '../utils/api.js';
import { makeCollapsible } from '../utils/panel-collapse.js';

let toolsContainer = null;

export function initToolsPanel() {
  // Create tools panel element
  const mainContainer = document.querySelector('.main-container');

  const toolsPanel = document.createElement('div');
  toolsPanel.className = 'tools-panel';
  toolsPanel.innerHTML = `
    <div class="panel-header">
      <span class="panel-title">Available Tools</span>
    </div>
    <div id="tools-list" class="tools-list">
      <div class="empty-tools">Loading tools...</div>
    </div>
  `;

  // Insert before bilibili panel
  const bilibiliPanel = document.getElementById('bilibili-panel');
  if (bilibiliPanel) {
    mainContainer.insertBefore(toolsPanel, bilibiliPanel);
  } else {
    mainContainer.appendChild(toolsPanel);
  }

  toolsContainer = document.getElementById('tools-list');

  // Make panel collapsible
  makeCollapsible(toolsPanel, 'tools', false);

  loadTools();
}

export async function loadTools() {
  try {
    const response = await apiCall('/ui/tools');
    renderTools(response.tools);
  } catch (e) {
    console.error('Error loading tools:', e);
    toolsContainer.innerHTML = '<div class="empty-tools">Failed to load tools</div>';
  }
}

function renderTools(tools) {
  if (!tools || tools.length === 0) {
    toolsContainer.innerHTML = '<div class="empty-tools">No tools available</div>';
    return;
  }

  toolsContainer.innerHTML = '';

  tools.forEach(tool => {
    const toolItem = document.createElement('div');
    toolItem.className = 'tool-item';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'tool-checkbox';
    checkbox.id = `tool-${tool.name}`;
    checkbox.checked = tool.enabled;
    checkbox.onchange = () => handleToolToggle(tool.name, checkbox.checked);

    const label = document.createElement('label');
    label.className = 'tool-label';
    label.htmlFor = `tool-${tool.name}`;

    const nameSpan = document.createElement('span');
    nameSpan.className = 'tool-name';
    nameSpan.textContent = tool.name;

    const descSpan = document.createElement('span');
    descSpan.className = 'tool-description';
    descSpan.textContent = tool.description;

    label.appendChild(nameSpan);
    label.appendChild(descSpan);

    toolItem.appendChild(checkbox);
    toolItem.appendChild(label);

    toolsContainer.appendChild(toolItem);
  });
}

async function handleToolToggle(name, enabled) {
  try {
    await apiCall('/ui/tools/toggle', {
      method: 'POST',
      body: { name, enabled }
    });
    console.log(`Tool ${name} ${enabled ? 'enabled' : 'disabled'}`);
  } catch (e) {
    console.error(`Error toggling tool ${name}:`, e);
    loadTools(); // Reload to reset UI state
  }
}
