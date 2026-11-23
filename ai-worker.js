// PRESSURELESS on-device AI worker.
// Model download, tokenization, and inference all run off the main UI thread.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
env.useWasmCache = true;
env.cacheKey = 'pressureless-transformers-cache-v1';

const PHONE_MODEL = 'onnx-community/SmolLM2-135M-Instruct-ONNX-MHA';
const DESKTOP_MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct';
let generator = null;
let loadingPromise = null;
let loadedLabel = '';

function send(type, extra = {}) {
  self.postMessage({ type, ...extra });
}

async function loadModel(isSmallDevice) {
  if (generator) return generator;
  if (loadingPromise) return loadingPromise;

  const modelId = isSmallDevice ? PHONE_MODEL : DESKTOP_MODEL;
  loadedLabel = isSmallDevice ? 'SmolLM2 135M (phone)' : 'SmolLM2 360M';

  loadingPromise = (async () => {
    send('status', { message: 'Preparing ' + loadedLabel + ' in the background…' });
    generator = await pipeline('text-generation', modelId, {
      device: 'wasm',
      dtype: 'q4',
      progress_callback(event) {
        if (event.status === 'progress' && event.total) {
          const percent = Math.max(0, Math.min(100, Math.round(event.progress || 0)));
          send('status', { message: 'Loading ' + loadedLabel + ' · ' + percent + '%' });
        } else if (event.status === 'done') {
          send('status', { message: 'Finishing ' + loadedLabel + '…' });
        }
      }
    });
    send('ready', { model: loadedLabel });
    return generator;
  })().catch(error => {
    generator = null;
    loadingPromise = null;
    send('load-error', { error: error?.message || String(error) });
    throw error;
  });

  return loadingPromise;
}

function extractAssistantText(output) {
  const generated = output?.[0]?.generated_text;
  if (Array.isArray(generated)) {
    for (let i = generated.length - 1; i >= 0; i--) {
      const message = generated[i];
      if (message?.role === 'assistant' && typeof message.content === 'string') {
        return message.content.trim();
      }
    }
  }
  if (typeof generated === 'string') {
    return generated
      .replace(/^\s*<\|im_start\|>assistant\s*/i, '')
      .replace(/<\|im_end\|>[\s\S]*$/i, '')
      .trim();
  }
  return '';
}

self.onmessage = async event => {
  const data = event.data || {};

  if (data.type === 'load') {
    try {
      await loadModel(!!data.smallDevice);
    } catch {
      // load-error is already sent by loadModel.
    }
    return;
  }

  if (data.type === 'generate') {
    const id = data.id;
    try {
      const model = await loadModel(!!data.smallDevice);
      const messages = Array.isArray(data.messages) ? data.messages : [];
      if (!messages.length) throw new Error('Please enter a question first.');

      send('status', { message: 'Generating your answer in the background…' });
      const output = await model(messages, {
        max_new_tokens: Math.max(48, Math.min(128, Number(data.maxTokens) || 88)),
        do_sample: false,
        repetition_penalty: 1.05
      });
      const text = extractAssistantText(output);
      if (!text) throw new Error('The model returned an empty answer. Try a shorter question.');
      send('result', { id, text });
    } catch (error) {
      send('generation-error', { id, error: error?.message || String(error) });
    }
  }
};
