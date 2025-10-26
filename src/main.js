import './style.css'
import { pipeline, env } from '@huggingface/transformers'

// 简易界面
const app = document.querySelector('#app')
app.innerHTML = `
  <div style="display:flex; flex-direction:column; gap:12px; align-items:center;">
    <h2 style="margin:0;">Whisper (tiny.en) 屏幕音频转字幕</h2>
    <div id="status" style="opacity:.8;">初始化…</div>
    <div id="subs" style="width:100%; max-width:960px; height:75vh; border:1px solid #555; padding:12px; border-radius:8px; text-align:left; overflow-y:auto; background:rgba(0,0,0,.15); scroll-behavior:smooth; word-break:break-word; overflow-wrap:anywhere;"></div>
  </div>
`
const $status = document.getElementById('status')
const $subs = document.getElementById('subs')

// 将字幕容器滚动到底部（考虑浏览器布局延迟）
function scrollToBottom(el) {
  try { el.scrollTop = el.scrollHeight } catch {}
  // 下一帧再滚动一次，确保布局完成
  requestAnimationFrame(() => {
    try { el.scrollTop = el.scrollHeight } catch {}
  })
}

;(async () => {
  try {
    // 1) 如支持，优先启用 WebGPU
    const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator
    try {
      if (env && env.backends && env.backends.onnx) {
        // 配置 ONNX 运行时：线程数与设备偏好
        env.backends.onnx.wasm = env.backends.onnx.wasm || {}
        env.backends.onnx.numThreads = env.backends.onnx.numThreads || navigator?.hardwareConcurrency || 4
        env.backends.onnx.devicePreference = hasWebGPU ? 'webgpu' : 'wasm'
      }
    } catch (e) {
      console.warn('配置 WebGPU 偏好失败，将自动回退到可用后端。', e)
    }

    // 2) 页面打开立刻请求屏幕 + 音频权限（会弹出权限提示）
    $status.textContent = '请求屏幕与系统音频权限…'
    const streamPromise = navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        // Chromium 的非标准约束，若不支持会被忽略
        // @ts-ignore
        systemAudio: 'include',
        // @ts-ignore
        suppressLocalAudioPlayback: false,
      }
    })
    // const model = 'Xenova/whisper-tiny.en'
    const model = 'Xenova/whisper-base.en'
    // 3) 同时开始加载 ASR 管线（并行）
    $status.textContent = `加载模型中（${model}）…首次加载较慢，请稍候`
    const asrPromise = pipeline(
      'automatic-speech-recognition',
      model,
      {
        device: hasWebGPU ? 'webgpu' : 'wasm',
      }
    )

    // 等待媒体流与模型都就绪
    const [stream, asr] = await Promise.all([streamPromise, asrPromise])

    if (!stream.getAudioTracks || stream.getAudioTracks().length === 0) {
      $status.textContent = '未检测到音频轨道。请在共享屏幕时勾选“共享系统音频”。'
    } else {
      $status.textContent = hasWebGPU ? '已获取音频流，使用 WebGPU 识别…' : '已获取音频流，使用 WASM 识别…'
    }

    // 4) 使用 WebAudio 捕获音频并维护缓冲
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)()
    const src = audioCtx.createMediaStreamSource(stream)
    const processor = audioCtx.createScriptProcessor(4096, 1, 1)

    // 为避免回声，不直接输出到扬声器，而是接入静音增益节点
    const silentGain = audioCtx.createGain()
    silentGain.gain.value = 0

    src.connect(processor)
    processor.connect(silentGain)
    silentGain.connect(audioCtx.destination)

    const inSampleRate = audioCtx.sampleRate // 常见为 48000Hz

    // 滑动窗口配置：3 秒窗口，1.5 秒步长
    const WINDOW_SEC = 3.0
    const HOP_SEC = 1.5
    const windowSamples = Math.floor(WINDOW_SEC * inSampleRate)
    const maxKeepSeconds = 12 // 额外多留些缓冲，避免抖动
    const maxKeepSamples = Math.floor(maxKeepSeconds * inSampleRate)

    let pcmBuffer = new Float32Array(0)
    let running = true
    let inferInProgress = false

    // 采集回调：将每帧 PCM 追加到环形缓冲
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

    // 线性重采样到 16k（Whisper 期望采样率）
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

    // 定时执行推理：每 1.5 秒取最近 3 秒进行识别
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
          scrollToBottom($subs)
          lastPrinted = text
        }

        $status.textContent = hasWebGPU ? '识别中（WebGPU）' : '识别中（WASM）'
      } catch (err) {
        console.error('识别出错：', err)
        $status.textContent = `识别出错：${err?.message || err}`
      } finally {
        inferInProgress = false
      }
    }, hopMs)

    // 资源清理（页面关闭或刷新时）
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
