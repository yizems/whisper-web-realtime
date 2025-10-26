import './style.css'
import { pipeline, env } from '@huggingface/transformers'

// Minimal UI
const app = document.querySelector('#app')
app.innerHTML = `
  <div style="display:flex; flex-direction:column; gap:12px; align-items:center;">
    <h2 style="margin:0;">Whisper (tiny.en) 屏幕音频转字幕</h2>
    <div id="status" style="opacity:.8;">初始化…</div>
    <div id="subs" style="width:100%; max-width:960px; min-height:120px; border:1px solid #555; padding:12px; border-radius:8px; text-align:left; overflow:auto; background:rgba(0,0,0,.15);"></div>
  </div>
`
const $status = document.getElementById('status')
const $subs = document.getElementById('subs')

;(async () => {
  try {
    // 1) Prefer WebGPU if available
    const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator
    try {
      if (env && env.backends && env.backends.onnx) {
        env.backends.onnx.wasm = env.backends.onnx.wasm || {}
        env.backends.onnx.numThreads = env.backends.onnx.numThreads || navigator?.hardwareConcurrency || 4
        env.backends.onnx.devicePreference = hasWebGPU ? 'webgpu' : 'wasm'
      }
    } catch (e) {
      console.warn('Failed to configure env for WebGPU, will fallback automatically.', e)
    }

    // 2) Immediately request screen + audio (triggers permission prompt on load)
    $status.textContent = '请求屏幕与系统音频权限…'
    const streamPromise = navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        // Non-standard (Chromium); ignored if unsupported
        // @ts-ignore
        systemAudio: 'include',
        // @ts-ignore
        suppressLocalAudioPlayback: false,
      }
    })

    // 3) Start loading ASR pipeline in parallel
    $status.textContent = '加载模型中（Xenova/whisper-tiny.en）…首次加载较慢，请稍候'
    const asrPromise = pipeline(
      'automatic-speech-recognition',
      'Xenova/whisper-tiny.en',
      {
        device: hasWebGPU ? 'webgpu' : 'wasm',
      }
    )

    // Wait for both stream and model
    const [stream, asr] = await Promise.all([streamPromise, asrPromise])

    if (!stream.getAudioTracks || stream.getAudioTracks().length === 0) {
      $status.textContent = '未检测到音频轨道。请在共享屏幕时勾选“共享系统音频”。'
    } else {
      $status.textContent = hasWebGPU ? '已获取音频流，使用 WebGPU 识别…' : '已获取音频流，使用 WASM 识别…'
    }

    // 4) WebAudio capture and buffer
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
    const src = audioCtx.createMediaStreamSource(stream)
    const processor = audioCtx.createScriptProcessor(4096, 1, 1)

    // Avoid echo: route into silent gain
    const silentGain = audioCtx.createGain()
    silentGain.gain.value = 0

    src.connect(processor)
    processor.connect(silentGain)
    silentGain.connect(audioCtx.destination)

    const inSampleRate = audioCtx.sampleRate // often 48000

    // Sliding window config
    const WINDOW_SEC = 3.0
    const HOP_SEC = 1.5
    const windowSamples = Math.floor(WINDOW_SEC * inSampleRate)
    const maxKeepSeconds = 12
    const maxKeepSamples = Math.floor(maxKeepSeconds * inSampleRate)

    let pcmBuffer = new Float32Array(0)
    let running = true
    let inferInProgress = false

    // Capture callback
    processor.onaudioprocess = (e) => {
      const ch0 = e.inputBuffer.getChannelData(0)
      const merged = new Float32Array(pcmBuffer.length + ch0.length)
      merged.set(pcmBuffer, 0)
      merged.set(ch0, pcmBuffer.length)
      pcmBuffer = merged
      if (pcmBuffer.length > maxKeepSamples) {
        pcmBuffer = pcmBuffer.slice(pcmBuffer.length - maxKeepSamples)
      }
    }

    // Linear resample to 16k (Whisper expected)
    function linearResample(input, inRate, outRate) {
      if (inRate === outRate) return input.slice(0)
      const ratio = outRate / inRate
      const outLength = Math.floor(input.length * ratio)
      const out = new Float32Array(outLength)
      for (let i = 0; i < outLength; i++) {
        const t = i / ratio
        const i0 = Math.floor(t)
        const i1 = Math.min(i0 + 1, input.length - 1)
        const frac = t - i0
        out[i] = input[i0] * (1 - frac) + input[i1] * frac
      }
      return out
    }

    // Periodic inference with sliding 3s window, 1.5s hop
    const hopMs = Math.floor(HOP_SEC * 1000)
    let lastPrinted = ''

    const inferTimer = setInterval(async () => {
      if (!running) return
      if (inferInProgress) return
      try {
        if (pcmBuffer.length < windowSamples) return
        inferInProgress = true
        const start = pcmBuffer.length - windowSamples
        const windowData = pcmBuffer.slice(start, start + windowSamples)
        const resampled = linearResample(windowData, inSampleRate, 16000)

        const result = await asr(resampled, {
          sampling_rate: 16000,
        })

        const text = (result && result.text) ? result.text.trim() : ''
        if (text && text !== lastPrinted) {
          const line = document.createElement('div')
          line.className = 'subtitle'
          line.textContent = text
          $subs.appendChild(line)
          $subs.scrollTop = $subs.scrollHeight
          lastPrinted = text
        }

        $status.textContent = hasWebGPU ? '识别中（WebGPU）' : '识别中（WASM）'
      } catch (err) {
        console.error('ASR error:', err)
        $status.textContent = `识别出错：${err?.message || err}`
      } finally {
        inferInProgress = false
      }
    }, hopMs)

    // Clean-up
    const cleanup = () => {
      running = false
      clearInterval(inferTimer)
      try { processor.disconnect(); } catch {}
      try { silentGain.disconnect(); } catch {}
      try { src.disconnect(); } catch {}
      try { audioCtx.close(); } catch {}
      stream.getTracks().forEach(t => t.stop())
    }
    window.addEventListener('beforeunload', cleanup)

  } catch (err) {
    console.error(err)
    $status.textContent = `初始化失败：${err?.message || err}`
  }
})()
