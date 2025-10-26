import './style.css'
import viteLogo from '/vite.svg'

document.querySelector('#app').innerHTML = `
  <div>
  demo
  </div>
`

navigator.mediaDevices.getDisplayMedia({
  video: true,
  audio: true,
}).then((stream) => {
  console.log(stream);
  console.log(stream.getAudioTracks());
  const audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);

  source.connect(processor);
  processor.connect(audioCtx.destination);

  processor.onaudioprocess = (e) => {
    const pcmData = e.inputBuffer.getChannelData(0); // Float32Array
    // 发送给 Whisper 模型
    console.log(pcmData)
  };
}).catch(e => {
  console.log(e)
});
