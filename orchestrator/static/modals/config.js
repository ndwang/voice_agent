/**
 * Configuration Editor Modal
 */

import { elements } from '../utils/dom.js';
import { state } from '../utils/state.js';
import { apiCall } from '../utils/api.js';
import { showStatusMessage } from '../utils/helpers.js';

// Store capabilities for hot-reload detection
let configCapabilities = {};

export const CONFIG_SCHEMA = {
  orchestrator: {
    title: 'Orchestrator',
    fields: {
      host: { type: 'text', label: 'Host' },
      port: { type: 'number', label: 'Port' },
      stt_websocket_path: { type: 'text', label: 'STT WebSocket Path' },
      log_level: { type: 'select', label: 'Log Level', options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'] },
      log_file: { type: 'text', label: 'Log File (null for none)' },
      enable_latency_tracking: { type: 'boolean', label: 'Enable Latency Tracking' },
      system_prompt_file: { type: 'text', label: 'System Prompt File' },
      hotkeys: {
        type: 'nested',
        label: 'Hotkeys',
        fields: {
          toggle_listening: { type: 'text', label: 'Toggle Listening' },
          cancel_speech: { type: 'text', label: 'Cancel Speech' }
        }
      }
    }
  },
  stt: {
    title: 'STT (Speech-to-Text)',
    fields: {
      host: { type: 'text', label: 'Host' },
      port: { type: 'number', label: 'Port' },
      provider: { type: 'provider_select', label: 'Provider', providers: ['funasr', 'faster-whisper'] },
      language_code: { type: 'text', label: 'Language Code' },
      sample_rate: { type: 'number', label: 'Sample Rate' },
      interim_transcript_min_samples: { type: 'number', label: 'Interim Transcript Min Samples' },
      log_level: { type: 'select', label: 'Log Level', options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'] },
      log_file: { type: 'text', label: 'Log File (null for none)' }
    },
    provider_fields: {
      'faster-whisper': {
        model_path: { type: 'text', label: 'Model Path' },
        device: { type: 'text', label: 'Device (null for auto)' },
        compute_type: { type: 'text', label: 'Compute Type (null for auto)' }
      },
      'funasr': {
        model_name: { type: 'text', label: 'Model Name' },
        vad_model: { type: 'text', label: 'VAD Model' },
        vad_kwargs: {
          type: 'nested',
          label: 'VAD Arguments',
          fields: {
            max_single_segment_time: { type: 'number', label: 'Max Single Segment Time (ms)' }
          }
        },
        punc_model: { type: 'text', label: 'Punctuation Model' },
        device: { type: 'text', label: 'Device' },
        batch_size_s: { type: 'number', label: 'Batch Size (seconds)' },
        streaming: {
          type: 'nested',
          label: 'Streaming Settings',
          fields: {
            enabled: { type: 'boolean', label: 'Enabled' },
            chunk_size: { type: 'text', label: 'Chunk Size (comma-separated array)' },
            encoder_chunk_look_back: { type: 'number', label: 'Encoder Chunk Look Back' },
            decoder_chunk_look_back: { type: 'number', label: 'Decoder Chunk Look Back' },
            vad_chunk_size_ms: { type: 'number', label: 'VAD Chunk Size (ms)' },
            silence_threshold_ms: { type: 'number', label: 'Silence Threshold (ms)' }
          }
        }
      }
    }
  },
  llm: {
    title: 'LLM (Language Model)',
    fields: {
      provider: { type: 'provider_select', label: 'Provider', providers: ['ollama', 'gemini'] }
    },
    provider_fields: {
      gemini: {
        model: { type: 'text', label: 'Model' },
        api_key: { type: 'password', label: 'API Key' },
        generation_config: {
          type: 'nested',
          label: 'Generation Config',
          fields: {
            thinking_budget: { type: 'number', label: 'Thinking Budget' }
          }
        }
      },
      ollama: {
        model: { type: 'text', label: 'Model' },
        base_url: { type: 'text', label: 'Base URL' },
        timeout: { type: 'number', label: 'Timeout (s)' },
        disable_thinking: { type: 'boolean', label: 'Disable Thinking' },
        generation_config: { type: 'text', label: 'Generation Config (null for none)' }
      }
    }
  },
  tts: {
    title: 'TTS (Text-to-Speech)',
    fields: {
      host: { type: 'text', label: 'Host' },
      port: { type: 'number', label: 'Port' },
      provider: { type: 'provider_select', label: 'Provider', providers: ['edge-tts', 'chattts', 'genie-tts', 'elevenlabs', 'gpt-sovits'] },
      log_level: { type: 'select', label: 'Log Level', options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'] },
      log_file: { type: 'text', label: 'Log File (null for none)' }
    },
    provider_fields: {
      'edge-tts': {
        voice: { type: 'text', label: 'Voice' },
        rate: { type: 'text', label: 'Rate' },
        pitch: { type: 'text', label: 'Pitch' }
      },
      'chattts': {
        model_source: { type: 'select', label: 'Model Source', options: ['local', 'huggingface', 'custom'] },
        device: { type: 'text', label: 'Device (null for auto)' }
      },
      'genie-tts': {
        character_name: { type: 'text', label: 'Character Name' },
        onnx_model_dir: { type: 'text', label: 'ONNX Model Dir' },
        language: { type: 'text', label: 'Language' },
        reference_audio_path: { type: 'text', label: 'Ref Audio Path' },
        reference_audio_text: { type: 'text', label: 'Ref Audio Text' },
        source_sample_rate: { type: 'number', label: 'Source Sample Rate' }
      },
      'elevenlabs': {
        voice_id: { type: 'text', label: 'Voice ID' },
        stability: { type: 'number', label: 'Stability (0-1)' },
        similarity_boost: { type: 'number', label: 'Similarity Boost (0-1)' },
        style: { type: 'number', label: 'Style (0-1)' }
      },
      'gpt-sovits': {
        server_url: { type: 'text', label: 'Server URL' },
        default_reference: { type: 'text', label: 'Default Reference' },
        default_text_lang: { type: 'text', label: 'Default Text Language' },
        gpt_weights_path: { type: 'text', label: 'GPT Weights Path' },
        sovits_weights_path: { type: 'text', label: 'SoVITS Weights Path' },
        streaming_mode: { type: 'number', label: 'Streaming Mode (0-3)' },
        temperature: { type: 'number', label: 'Temperature' },
        top_p: { type: 'number', label: 'Top P' },
        top_k: { type: 'number', label: 'Top K' },
        speed_factor: { type: 'number', label: 'Speed Factor' },
        timeout: { type: 'number', label: 'Timeout (seconds)' },
        references: {
          type: 'reference_dict',
          label: 'References',
          fields: {
            name: { type: 'text', label: 'Reference Name' },
            ref_audio_path: { type: 'text', label: 'Reference Audio Path' },
            prompt_text: { type: 'text', label: 'Prompt Text' },
            prompt_lang: { type: 'text', label: 'Prompt Language' }
          }
        }
      }
    }
  },
  ocr: {
    title: 'OCR',
    fields: {
      host: { type: 'text', label: 'Host' },
      port: { type: 'number', label: 'Port' },
      language: { type: 'text', label: 'Language' },
      interval_ms: { type: 'number', label: 'Interval (ms)' },
      texts_storage_file_prefix: { type: 'text', label: 'Texts Storage File Prefix' },
      log_level: { type: 'select', label: 'Log Level', options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'] },
      log_file: { type: 'text', label: 'Log File (null for none)' }
    }
  },
  audio: {
    title: 'Audio Settings',
    fields: {
      input: {
        type: 'nested',
        label: 'Input Settings',
        fields: {
          sample_rate: { type: 'number', label: 'Sample Rate' },
          channels: { type: 'number', label: 'Channels' },
          device: { type: 'text', label: 'Device Index (null for default)' }
        }
      },
      output: {
        type: 'nested',
        label: 'Output Settings',
        fields: {
          sample_rate: { type: 'number', label: 'Sample Rate' },
          channels: { type: 'number', label: 'Channels' },
          device: { type: 'text', label: 'Device Index (null for default)' }
        }
      },
      dtype: { type: 'text', label: 'Data Type' },
      block_size_ms: { type: 'number', label: 'Block Size (ms)' },
      listening_status_poll_interval: { type: 'number', label: 'Listening Status Poll Interval (s)' }
    }
  },
  bilibili: {
    title: 'Bilibili',
    fields: {
      enabled: { type: 'boolean', label: 'Enabled' },
      room_id: { type: 'number', label: 'Room ID' },
      sessdata: { type: 'password', label: 'SESSDATA' },
      danmaku_ttl_seconds: { type: 'number', label: 'Danmaku TTL (seconds)' }
    }
  },
  obs: {
    title: 'OBS',
    fields: {
      websocket: {
        type: 'nested',
        label: 'Websocket Settings',
        fields: {
          host: { type: 'text', label: 'Host' },
          port: { type: 'number', label: 'Port' },
          password: { type: 'password', label: 'Password' }
        }
      },
      subtitle_source: { type: 'text', label: 'Subtitle Source Name' },
      subtitle_ttl_seconds: { type: 'number', label: 'Subtitle TTL (seconds)' },
      visibility_source: { type: 'text', label: 'Visibility Source Name' },
      appear_filter_name: { type: 'text', label: 'Appear Filter Name' },
      clear_filter_name: { type: 'text', label: 'Clear Filter Name' }
    }
  },
  services: {
    title: 'Service URLs',
    fields: {
      orchestrator_base_url: { type: 'text', label: 'Orchestrator Base URL' },
      stt_websocket_url: { type: 'text', label: 'STT WebSocket URL' },
      tts_websocket_url: { type: 'text', label: 'TTS WebSocket URL' },
      ocr_websocket_url: { type: 'text', label: 'OCR WebSocket URL' },
      ocr_base_url: { type: 'text', label: 'OCR Base URL' }
    }
  }
};

function createReferenceBox(refName, refData, fieldSchema, parentPath) {
  const box = document.createElement('div');
  box.className = 'reference-box';

  // Create 4 fields in a grid: name, ref_audio_path, prompt_text, prompt_lang
  Object.entries(fieldSchema).forEach(([fieldKey, fieldDef]) => {
    const fieldGroup = document.createElement('div');
    fieldGroup.className = 'config-form-group';

    const label = document.createElement('label');
    label.className = 'config-label';
    label.textContent = fieldDef.label;
    fieldGroup.appendChild(label);

    const input = document.createElement('input');
    input.type = fieldDef.type === 'password' ? 'password' : 'text';
    input.className = 'config-input';
    input.dataset.refField = fieldKey;

    if (fieldKey === 'name') {
      input.value = refName || '';
    } else {
      input.value = (refData && refData[fieldKey]) || '';
    }

    fieldGroup.appendChild(input);
    box.appendChild(fieldGroup);
  });

  // Delete button
  const deleteBtn = document.createElement('button');
  deleteBtn.type = 'button';
  deleteBtn.className = 'delete-reference-btn';
  deleteBtn.textContent = '× Remove';
  deleteBtn.onclick = () => box.remove();
  box.appendChild(deleteBtn);

  return box;
}

function createFormField(key, field, value, path = []) {
  const fullPath = [...path, key].join('.');
  const group = document.createElement('div');
  group.className = 'config-form-group';

  if (field.type === 'nested') {
    const nestedGroup = document.createElement('div');
    nestedGroup.className = 'nested-group';
    const title = document.createElement('div');
    title.className = 'nested-title';
    title.textContent = field.label;
    nestedGroup.appendChild(title);

    Object.entries(field.fields).forEach(([subKey, subField]) => {
      const subValue = (value && typeof value === 'object') ? value[subKey] : undefined;
      nestedGroup.appendChild(createFormField(subKey, subField, subValue, [...path, key]));
    });
    return nestedGroup;
  }

  if (field.type === 'reference_dict') {
    const refContainer = document.createElement('div');
    refContainer.className = 'reference-dict-container';

    const title = document.createElement('div');
    title.className = 'nested-title';
    title.textContent = field.label;
    refContainer.appendChild(title);

    const refsWrapper = document.createElement('div');
    refsWrapper.className = 'references-wrapper';
    refsWrapper.dataset.path = fullPath;

    // Display existing references
    if (value && typeof value === 'object') {
      Object.entries(value).forEach(([refName, refData]) => {
        const refBox = createReferenceBox(refName, refData, field.fields, fullPath);
        refsWrapper.appendChild(refBox);
      });
    }

    refContainer.appendChild(refsWrapper);

    // Add button to add new reference
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'add-reference-btn';
    addBtn.textContent = '+ Add Reference';
    addBtn.onclick = () => {
      const refBox = createReferenceBox('', {}, field.fields, fullPath);
      refsWrapper.appendChild(refBox);
    };
    refContainer.appendChild(addBtn);

    return refContainer;
  }
  
  const label = document.createElement('label');
  label.className = 'config-label';
  label.textContent = field.label;
  group.appendChild(label);
  
  let input;
  if (field.type === 'select' || field.type === 'provider_select') {
    input = document.createElement('select');
    input.className = 'config-input';
    const options = field.type === 'provider_select' ? field.providers : field.options;
    options.forEach(opt => {
      const option = document.createElement('option');
      option.value = opt;
      option.textContent = opt;
      if (opt === value) option.selected = true;
      input.appendChild(option);
    });
    
    if (field.type === 'provider_select') {
      input.dataset.isProviderSelect = 'true';
      input.dataset.section = path[0] || key;
    }
  } else if (field.type === 'boolean') {
    group.className = 'config-form-group config-checkbox-group';
    input = document.createElement('input');
    input.type = 'checkbox';
    input.className = 'config-checkbox';
    input.checked = !!value;
    group.innerHTML = '';
    group.appendChild(input);
    group.appendChild(label);
  } else {
    input = document.createElement('input');
    input.type = field.type === 'password' ? 'password' : field.type === 'number' ? 'number' : 'text';
    input.className = 'config-input';
    // Handle array values by converting to comma-separated string
    if (Array.isArray(value)) {
      input.value = value.join(', ');
    } else {
      input.value = value === null || value === undefined ? '' : value;
    }
  }
  
  input.dataset.path = fullPath;
  input.name = fullPath;

  // Add change listener for restart warnings
  input.addEventListener('change', () => checkRestartRequired(input, fullPath));

  group.appendChild(input);

  return group;
}

function checkRestartRequired(input, fullPath) {
  // Check if this field requires restart
  const capability = configCapabilities[fullPath];

  if (capability && capability.hot_reload === false) {
    // Show immediate warning
    showStatusMessage(
      elements.configStatusMessage,
      `⚠️ ${fullPath} requires a service restart to take effect`,
      true,
      5000
    );
  }
}

function buildAccordion() {
  elements.configAccordion.innerHTML = '';
  
  Object.entries(CONFIG_SCHEMA).forEach(([sectionKey, section]) => {
    const item = document.createElement('div');
    item.className = 'accordion-item';
    
    const header = document.createElement('div');
    header.className = 'accordion-header';
    header.textContent = section.title;
    header.onclick = () => {
      item.classList.toggle('active');
    };
    
    const content = document.createElement('div');
    content.className = 'accordion-content';
    content.id = `section-${sectionKey}`;
    
    const sectionData = state.currentConfig[sectionKey] || {};
    Object.entries(section.fields).forEach(([fieldKey, field]) => {
      const fieldElement = createFormField(fieldKey, field, sectionData[fieldKey], [sectionKey]);
      content.appendChild(fieldElement);
      
      if (field.type === 'provider_select') {
        const providerSelect = fieldElement.querySelector('select');
        providerSelect.onchange = (e) => {
          updateProviderFields(sectionKey, e.target.value);
        };
      }
    });
    
    if (section.provider_fields) {
      const providerContainer = document.createElement('div');
      providerContainer.className = 'provider-section';
      providerContainer.id = `provider-fields-${sectionKey}`;
      content.appendChild(providerContainer);
    }
    
    item.appendChild(header);
    item.appendChild(content);
    elements.configAccordion.appendChild(item);
  });
}

function updateProviderFields(sectionKey, providerKey) {
  const container = document.getElementById(`provider-fields-${sectionKey}`);
  if (!container) return;
  
  container.innerHTML = '';
  const sectionSchema = CONFIG_SCHEMA[sectionKey];
  if (!sectionSchema?.provider_fields?.[providerKey]) {
    return;
  }
  
  const providerSchema = sectionSchema.provider_fields[providerKey];
  const sectionData = state.currentConfig[sectionKey] || {};
  const providersData = sectionData.providers || {};
  const providerData = providersData[providerKey] || {};
  
  Object.entries(providerSchema).forEach(([fieldKey, field]) => {
    container.appendChild(createFormField(fieldKey, field, providerData[fieldKey], [sectionKey, 'providers', providerKey]));
  });
}

function setDeepValue(obj, path, value) {
  const parts = path.split('.');
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (!(part in current)) current[part] = {};
    current = current[part];
  }
  current[parts[parts.length - 1]] = value;
}

export async function loadFullConfig() {
  try {
    // Load capabilities first
    try {
      configCapabilities = await apiCall('/api/config/capabilities');
    } catch (e) {
      console.warn('Could not load config capabilities:', e);
      configCapabilities = {};
    }

    state.currentConfig = await apiCall('/ui/config');
    buildAccordion();

    Object.entries(CONFIG_SCHEMA).forEach(([sectionKey]) => {
      const providerSelect = document.querySelector(`select[data-is-provider-select="true"][data-section="${sectionKey}"]`);
      if (providerSelect) {
        updateProviderFields(sectionKey, providerSelect.value);
      }
    });
  } catch (e) {
    console.error('Error loading config:', e);
    showStatusMessage(elements.configStatusMessage, 'Failed to load configuration', true, 5000);
  }
}

function getDeepValue(obj, path) {
  const parts = path.split('.');
  let current = obj;
  for (let i = 0; i < parts.length; i++) {
    if (current === null || current === undefined) return undefined;
    current = current[parts[i]];
  }
  return current;
}

function inferType(value, originalValue) {
  // If the input is empty, return null
  if (value === '' || value === 'null') {
    return null;
  }

  // If we have an original value, try to preserve its type
  if (originalValue !== undefined && originalValue !== null) {
    // Array type - check if original was an array
    if (Array.isArray(originalValue)) {
      // Parse comma-separated values into array
      if (typeof value === 'string') {
        return value.split(',').map(v => {
          const trimmed = v.trim();
          // Try to parse as number if original array contained numbers
          if (!isNaN(trimmed) && trimmed !== '') {
            return trimmed.includes('.') ? parseFloat(trimmed) : parseInt(trimmed, 10);
          }
          return trimmed;
        });
      }
      return originalValue; // Keep original if can't parse
    }

    // Number type - check if original was a number
    if (typeof originalValue === 'number') {
      const num = parseFloat(value);
      return isNaN(num) ? value : num;
    }

    // Boolean type - check if original was boolean
    if (typeof originalValue === 'boolean') {
      return value === true || value === 'true';
    }

    // Object type - try to parse as JSON if it's a string
    if (typeof originalValue === 'object' && typeof value === 'string') {
      try {
        return JSON.parse(value);
      } catch {
        return value;
      }
    }

    // For strings, keep as string (don't try to convert to numbers)
    // The schema's input type should handle number conversion
    return value;
  }

  // No original value - just return the string as-is
  // The schema's input type should determine if it needs to be a number
  return value;
}

export async function saveFullConfig() {
  const newConfig = JSON.parse(JSON.stringify(state.currentConfig));

  // First, handle reference dictionaries separately
  const refWrappers = elements.configAccordion.querySelectorAll('.references-wrapper');
  refWrappers.forEach(wrapper => {
    const path = wrapper.dataset.path;
    if (!path) return;

    const references = {};
    const refBoxes = wrapper.querySelectorAll('.reference-box');

    refBoxes.forEach(box => {
      const inputs = box.querySelectorAll('input');
      let refName = '';
      const refData = {};

      inputs.forEach(input => {
        const fieldKey = input.dataset.refField;
        if (fieldKey === 'name') {
          refName = input.value.trim();
        } else {
          refData[fieldKey] = input.value || null;
        }
      });

      if (refName) {
        references[refName] = refData;
      }
    });

    setDeepValue(newConfig, path, references);
  });

  // Then handle regular inputs (but skip inputs inside reference boxes)
  const allInputs = elements.configAccordion.querySelectorAll('input:not(.reference-box input), select');
  allInputs.forEach(input => {
    const path = input.dataset.path;
    if (!path) return;

    let val;
    if (input.type === 'checkbox') {
      val = input.checked;
    } else if (input.type === 'password') {
      // For password fields, keep the value as-is (string or null)
      val = input.value || null;
    } else if (input.type === 'number') {
      // For number inputs, always convert to number
      val = input.value === '' ? null : parseFloat(input.value);
    } else {
      val = input.value;
      // Get the original value to preserve type
      const originalValue = getDeepValue(state.currentConfig, path);

      // Use type inference based on original value
      val = inferType(val, originalValue);
    }

    setDeepValue(newConfig, path, val);
  });

  elements.configModalSave.disabled = true;
  try {
    const response = await apiCall('/ui/config', {
      method: 'POST',
      body: { config: newConfig },
    });

    // Check if response includes restart requirements
    let message = '✓ Configuration saved successfully!';

    if (response.needs_restart && response.needs_restart.length > 0) {
      message += '\n\n⚠️ Restart required for:\n';
      message += response.needs_restart.map(item => `  • ${item}`).join('\n');
    } else {
      message += ' All changes applied immediately.';
    }

    showStatusMessage(elements.configStatusMessage, message, response.needs_restart && response.needs_restart.length > 0, 8000);
    state.currentConfig = newConfig;
  } catch (e) {
    console.error('Error saving config:', e);
    showStatusMessage(elements.configStatusMessage, e.message, true, 5000);
  } finally {
    elements.configModalSave.disabled = false;
  }
}

export function closeConfigModal() {
  elements.configModal.classList.remove('active');
}

export function setupConfigModal() {
  elements.configBtn.onclick = () => {
    elements.configModal.classList.add('active');
    loadFullConfig();
  };
  elements.configModalClose.onclick = closeConfigModal;
  elements.configModalCancel.onclick = closeConfigModal;
  elements.configModalSave.onclick = saveFullConfig;
  elements.configModal.onclick = (e) => {
    if (e.target === elements.configModal) {
      closeConfigModal();
    }
  };
}

