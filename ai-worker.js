// PRESSURELESS on-device AI worker.
// Downloads, tokenization, and generation stay off the UI thread.
import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
env.useWasmCache = true;
env.cacheKey = 'pressureless-transformers-cache-v2';

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
    const choices = [];
    let adapter = null;
    try {
      if (self.navigator?.gpu?.requestAdapter) {
        adapter = await self.navigator.gpu.requestAdapter();
      }
    } catch (error) {
      send('status', { message: 'Checking graphics acceleration…' });
    }

    if (adapter) {
      const supportsFloat16 = !!adapter.features?.has?.('shader-f16');
      choices.push({
        device: 'webgpu',
        dtype: supportsFloat16 ? 'q4f16' : 'q4',
        label: 'WebGPU accelerated'
      });
    }
    choices.push({ device: 'wasm', dtype: 'q4', label: 'CPU compatibility mode' });

    let lastError = null;
    for (const choice of choices) {
      try {
        send('status', { message: 'Loading ' + loadedLabel + ' · ' + choice.label + '…' });
        generator = await pipeline('text-generation', modelId, {
          device: choice.device,
          dtype: choice.dtype,
          progress_callback(event) {
            if (event.status === 'progress_total') {
              const percent = Math.max(0, Math.min(100, Math.round((event.progress || 0) * 100)));
              send('status', { message: 'Downloading ' + loadedLabel + ' · ' + percent + '%' });
            } else if (event.status === 'progress' && event.total) {
              const loaded = Number(event.loaded || 0);
              const total = Number(event.total || 0);
              const percent = Math.max(0, Math.min(100, Math.round(event.progress || (total ? loaded / total * 100 : 0))));
              send('status', { message: 'Downloading ' + loadedLabel + ' · ' + percent + '%' });
            } else if (event.status === 'initiate') {
              send('status', { message: 'Connecting to the model files…' });
            } else if (event.status === 'done') {
              send('status', { message: 'Preparing the ' + loadedLabel + ' runtime…' });
            }
          }
        });
        send('ready', { model: loadedLabel, device: choice.label, dtype: choice.dtype });
        return generator;
      } catch (error) {
        lastError = error;
        generator = null;
        if (choice.device === 'webgpu') {
          send('status', { message: 'Graphics acceleration unavailable; switching to compatibility mode…' });
        }
      }
    }
    throw lastError || new Error('No supported inference engine is available in this browser.');
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
      if (message?.role === 'assistant' && typeof message.content === 'string') return message.content.trim();
    }
  }
  if (typeof generated === 'string') {
    return generated.replace(/^\s*<\|im_start\|>assistant\s*/i, '').replace(/<\|im_end\|>[\s\S]*$/i, '').trim();
  }
  return '';
}

self.onmessage = async event => {
  const data = event.data || {};
  if (data.type === 'load') {
    try { await loadModel(!!data.smallDevice); }
    catch { /* load-error has already been sent */ }
    return;
  }
  if (data.type === 'generate') {
    const id = data.id;
    try {
      const model = await loadModel(!!data.smallDevice);
      const messages = Array.isArray(data.messages) ? data.messages : [];
      if (!messages.length) throw new Error('Please enter a question first.');
      send('status', { message: 'Generating your answer…' });
      const streamer = new TextStreamer(model.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: chunk => {
          if (chunk) send('token', { id, text: chunk });
        }
      });
      const output = await model(messages, {
        max_new_tokens: Math.max(40, Math.min(88, Number(data.maxTokens) || (data.smallDevice ? 56 : 80))),
        do_sample: false,
        repetition_penalty: 1.02,
        use_cache: true,
        streamer
      });
      const text = extractAssistantText(output);
      if (!text) throw new Error('The model returned an empty answer. Try a shorter question.');
      send('result', { id, text });
    } catch (error) {
      send('generation-error', { id, error: error?.message || String(error) });
    }
  }
};
