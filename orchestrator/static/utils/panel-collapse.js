/**
 * Panel Collapse Utility
 * Provides reusable collapse/expand functionality for side panels
 */

const STORAGE_KEY_PREFIX = 'panel-collapsed-';

/**
 * Make a panel collapsible/expandable
 *
 * @param {HTMLElement} panel - The panel element to make collapsible
 * @param {string} panelId - Unique ID for this panel (for localStorage)
 * @param {boolean} defaultCollapsed - Default collapsed state
 */
export function makeCollapsible(panel, panelId, defaultCollapsed = false) {
  // Get stored state or use default
  const storageKey = STORAGE_KEY_PREFIX + panelId;
  const isCollapsed = localStorage.getItem(storageKey) === 'true' ||
                     (localStorage.getItem(storageKey) === null && defaultCollapsed);

  // Find or create header
  let header = panel.querySelector('.panel-header');
  if (!header) {
    console.warn('Panel header not found for', panelId);
    return;
  }

  // Add collapse button
  const collapseBtn = document.createElement('button');
  collapseBtn.className = 'panel-collapse-btn';
  collapseBtn.innerHTML = isCollapsed ? '◀' : '▶';
  collapseBtn.title = isCollapsed ? 'Expand panel' : 'Collapse panel';
  collapseBtn.setAttribute('aria-label', isCollapsed ? 'Expand panel' : 'Collapse panel');

  // Insert button at the start of header
  header.insertBefore(collapseBtn, header.firstChild);

  // Apply initial state
  if (isCollapsed) {
    panel.classList.add('collapsed');
  }

  // Toggle handler
  collapseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const nowCollapsed = panel.classList.toggle('collapsed');

    // Update button
    collapseBtn.innerHTML = nowCollapsed ? '◀' : '▶';
    collapseBtn.title = nowCollapsed ? 'Expand panel' : 'Collapse panel';
    collapseBtn.setAttribute('aria-label', nowCollapsed ? 'Expand panel' : 'Collapse panel');

    // Save state
    localStorage.setItem(storageKey, nowCollapsed);
  });

  return {
    collapse: () => {
      panel.classList.add('collapsed');
      collapseBtn.innerHTML = '◀';
      collapseBtn.title = 'Expand panel';
      localStorage.setItem(storageKey, 'true');
    },
    expand: () => {
      panel.classList.remove('collapsed');
      collapseBtn.innerHTML = '▶';
      collapseBtn.title = 'Collapse panel';
      localStorage.setItem(storageKey, 'false');
    },
    isCollapsed: () => panel.classList.contains('collapsed')
  };
}
