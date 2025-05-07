<template>
  <div class="tool trainer-container">
    <h2>🧠 ONNX Image Classifier</h2>
    <input type="file" @change="handleFile" accept="image/*" />
    <canvas ref="canvas" style="display:none" width="224" height="224"></canvas>

    <div v-if="imageSrc" style="margin-top: 1rem;">
      <img :src="imageSrc" alt="Uploaded Image" style="max-width: 300px;" />
    </div>

    <p v-if="prediction">{{ prediction }}</p>
    <p v-if="error" style="color:red"><strong>Error:</strong> {{ error }}</p>
  </div>
  <h1>Image Classification</h1>
<h2>with ONNX Runtime Web</h2>
<div class="explanation">
  <h3>📖 What This Does</h3>
  <p>This demo uses an ONNX version of MobileNetV2 to classify images you upload. It runs entirely in the browser using the ONNX WebAssembly backend for fast, secure inference.</p>

  <h3>🌐 Real‑World Applications</h3>
  <ul>
    <li>Product tagging and image search</li>
    <li>Automated labeling for datasets</li>
    <li>On-device AI for privacy-preserving classification</li>
  </ul>

  <h3>🔧 Broader Uses for Vision Models</h3>
  <ul>
    <li><strong>Security:</strong> Detect objects in surveillance feeds</li>
    <li><strong>Retail:</strong> Visual search for matching items</li>
    <li><strong>Healthcare:</strong> Triage images or scans</li>
  </ul>

  <h3>\u{1f6e0️} Tech Details</h3>
  <p>Powered by ONNX Runtime Web. The MobileNetV2 model was pre-trained on ImageNet and optimized for edge inference. Only pixel data is processed locally without cloud calls.</p>
</div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const canvas = ref(null)
const prediction = ref('')
const error = ref(null)
const imageSrc = ref(null)
let session = null
const labels = ref([])

onMounted(async () => {
  try {
    await waitForOrt() // <-- Wait until ort is available

    const ort = window.ort

    // Configure WASM backend
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/"
    ort.env.wasm.useJsep = false
    ort.env.wasm.proxy = false
    ort.env.wasm.numThreads = 1

    // Load labels
    const res = await fetch("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json")
    labels.value = await res.json()

    // Load ONNX model
    session = await ort.InferenceSession.create("/mobilenetv2-7.onnx", {
      executionProviders: ["wasm"]
    })

    console.log("✅ ONNX model loaded with WASM backend (JSEP disabled)")
  } catch (err) {
    console.error("❌ ONNX init error:", err)
    error.value = String(err)
  }
})

async function waitForOrt(timeout = 3000) {
  const start = performance.now()
  while (!window.ort) {
    await new Promise(r => setTimeout(r, 50))
    if (performance.now() - start > timeout) {
      throw new Error("ort failed to load from CDN within timeout.")
    }
  }
}


const handleFile = async (e) => {
  const file = e.target.files[0]
  if (!file) return

  imageSrc.value = URL.createObjectURL(file)

  const img = new Image()
  img.onload = async () => {
    try {
      const ctx = canvas.value.getContext('2d', { willReadFrequently: true })
      ctx.clearRect(0, 0, 224, 224)
      ctx.drawImage(img, 0, 0, 224, 224)
      const inputTensor = getTensorFromCanvas(canvas.value)
      const feeds = { input: inputTensor }

      const results = await session.run(feeds)
      const output = results[session.outputNames[0]]
      const topIdx = output.data.indexOf(Math.max(...output.data))
      const label = labels.value[topIdx] || 'Unknown'
      prediction.value = `Prediction: ${label} (Class index: ${topIdx})`
    } catch (err) {
      console.error('❌ Prediction error:', err)
      error.value = String(err)
    }
  }
  img.src = imageSrc.value
}

function getTensorFromCanvas(canvas) {
  const ort = window.ort
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  const imageData = ctx.getImageData(0, 0, 224, 224)
  const input = new Float32Array(3 * 224 * 224)
  for (let i = 0; i < 224 * 224; i++) {
    input[i] = imageData.data[i * 4] / 255
    input[i + 224 * 224] = imageData.data[i * 4 + 1] / 255
    input[i + 2 * 224 * 224] = imageData.data[i * 4 + 2] / 255
  }
  return new ort.Tensor('float32', input, [1, 3, 224, 224])
}
</script>

<style scoped>

  .tool{
    padding: 4rem 0;
  }

</style>
