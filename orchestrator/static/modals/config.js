/**
 * Configuration Editor Modal
 */

import { elements } from '../utils/dom.js';
import { state } from '../utils/state.js';
import { apiCall } from '../utils/api.js';
import { showStatusMessage } from '../utils/helpers.js';

export const CONFIG_SCHEMA = {
  orchestrator: {
    title: 'Orchestrator',
    fields: {
      host: { type: 'text', label: 'Host' },
      port: { type: 'number', label: 'Port' },
      log_level: { type: 'select', label: 'Log Level', options: ['DEBUG', 'INFO', 'WARNING', 'ERROR'] },
      enable_latency_tracking: { type: 'boolean', label: 'Enable Latency Tracking' }
    }
  },
  stt: {
    title: 'STT (Speech-to-Text)',
    fields: {
      provider: { type: 'provider_select', label: 'Provider', providers: ['funasr', 'faster-whisper'] },
      language_code: { type: 'text', label: 'Language Code' },
      sample_rate: { type: 'number', label: 'Sample Rate' }
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
        punc_model: { type: 'text', label: 'Punctuation Model' },
        streaming: {
          type: 'nested',
          label: 'Streaming Settings',
          fields: {
            enabled: { type: 'boolean', label: 'Enabled' },
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
        api_key: { type: 'password', label: 'API Key' }
      },
      ollama: {
        model: { type: 'text', label: 'Model' },
        base_url: { type: 'text', label: 'Base URL' },
        timeout: { type: 'number', label: 'Timeout (s)' },
        disable_thinking: { type: 'boolean', label: 'Disable Thinking' }
      }
    }
  },
  tts: {
    title: 'TTS (Text-to-Speech)',
    fields: {
      provider: { type: 'provider_select', label: 'Provider', providers: ['edge-tts', 'chattts', 'genie-tts', 'elevenlabs', 'gpt-sovits'] },
      output_sample_rate: { type: 'number', label: 'Output Sample Rate (null for auto)' }
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
        source_sample_rate: { type: 'number', label: 'Source Sample Rate' }
      },
      'elevenlabs': {
        voice_id: { type: 'text', label: 'Voice ID' },
        stability: { type: 'number', label: 'Stability (0-1)' },
        similarity_boost: { type: 'number', label: 'Similarity Boost (0-1)' }
      },
      'gpt-sovits': {
        server_url: { type: 'text', label: 'Server URL' },
        default_reference: { type: 'text', label: 'Default Reference' },
        default_text_lang: { type: 'text', label: 'Default Text Language' },
        streaming_mode: { type: 'number', label: 'Streaming Mode (0-3)' },
        temperature: { type: 'number', label: 'Temperature' },
        top_p: { type: 'number', label: 'Top P' },
        top_k: { type: 'number', label: 'Top K' },
        speed_factor: { type: 'number', label: 'Speed Factor' },
        timeout: { type: 'number', label: 'Timeout (seconds)' }
      }
    }
  },
  ocr: {
    title: 'OCR',
    fields: {
      host: { type: 'text', label: 'Host' },
      port: { type: 'number', label: 'Port' },
      language: { type: 'text', label: 'Language' },
      interval_ms: { type: 'number', label: 'Interval (ms)' }
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
      silence_threshold_ms: { type: 'number', label: 'Silence Threshold (ms)' },
      vad_min_speech_prob: { type: 'number', label: 'VAD Min Speech Prob (0-1)' }
    }
  },
  bilibili: {
    title: 'Bilibili',
    fields: {
      enabled: { type: 'boolean', label: 'Enabled' },
      room_id: { type: 'number', label: 'Room ID' },
      sessdata: { type: 'password', label: 'SESSDATA' }
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
      subtitle_source: { type: 'text', label: 'Subtitle Source Name' }
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
    input.type = field.type === 'password' ? 'password' : 'text';
    input.className = 'config-input';
    input.value = value === null || value === undefined ? '' : value;
  }
  
  input.dataset.path = fullPath;
  input.name = fullPath;
  group.appendChild(input);
  
  return group;
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

export async function saveFullConfig() {
  const newConfig = JSON.parse(JSON.stringify(state.currentConfig));
  
  const allInputs = elements.configAccordion.querySelectorAll('input, select');
  allInputs.forEach(input => {
    const path = input.dataset.path;
    if (!path) return;
    
    let val;
    if (input.type === 'checkbox') {
      val = input.checked;
    } else {
      val = input.value;
      if (input.type === 'number' || (!isNaN(val) && val !== '' && input.type !== 'text' && input.type !== 'password')) {
        val = val.includes('.') ? parseFloat(val) : parseInt(val, 10);
      } else if (val === '' && input.type !== 'password') {
        val = null;
      }
    }
    
    setDeepValue(newConfig, path, val);
  });
  
  elements.configModalSave.disabled = true;
  try {
    await apiCall('/ui/config', {
      method: 'POST',
      body: { config: newConfig },
    });
    showStatusMessage(elements.configStatusMessage, 'Configuration saved successfully! Some changes may require a restart.', false, 5000);
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

