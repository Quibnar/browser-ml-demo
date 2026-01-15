<template>
  <DemoWrapper>
    <!-- TOOL SLOT -->
    <template #tool>
      <div v-if="loading" class="loading">
        <div class="loading-spinner"></div>
        <p>Loading model…</p>
      </div>
      <div v-else class="tool">
        <div class="file-input-container">
          <label for="image-upload" class="file-input-label">
            📁 Choose an image
          </label>
          <input 
            id="image-upload"
            type="file" 
            @change="handleFile" 
            accept="image/*" 
            class="file-input"
          />
        </div>
        
        <canvas ref="canvas" style="display:none" width="224" height="224"></canvas>

        <div v-if="imageSrc" class="image-preview">
          <img :src="imageSrc" alt="Uploaded Image" />
        </div>

        <div v-if="prediction" class="prediction-result">
          <h3>Classification Result:</h3>
          <p class="prediction-text">{{ prediction }}</p>
        </div>
        
        <div v-if="error" class="error">
          <h3>Error:</h3>
          <p>{{ error }}</p>
          <button @click="retryLoad" class="retry-button">🔄 Retry</button>
        </div>
      </div>
    </template>

    <!-- EXPLANATION SLOT -->
    <template #explanation>
      <h2>Image Classification</h2>
      <h3>📖 What This Does</h3>
      <p>
        This demo uses an ONNX version of MobileNetV2 to classify images you
        upload. It runs entirely in the browser using the ONNX WebAssembly
        backend for fast, secure inference.
      </p>

      <h3>🌐 Real-World Applications</h3>
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

      <h3>🛠️ Tech Details</h3>
      <p>
        Powered by ONNX Runtime Web. The MobileNetV2 model was pre-trained on
        ImageNet and optimized for edge inference. Only pixel data is processed
        locally without cloud calls.
      </p>
    </template>
  </DemoWrapper>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import DemoWrapper from './DemoWrapper.vue'

const canvas = ref(null)
const prediction = ref('')
const error = ref(null)
const imageSrc = ref(null)
const loading = ref(true)
let session = null
const labels = ref([])

const retryLoad = async () => {
  error.value = null
  loading.value = true
  await initializeModel()
}

const initializeModel = async () => {
  try {
    await waitForOrt()
    const ort = window.ort

    // Configure WASM backend with local paths
    ort.env.wasm.wasmPaths = '/ort/'
    ort.env.wasm.useJsep = false
    ort.env.wasm.proxy = false
    ort.env.wasm.numThreads = 1

    // Load labels
    const res = await fetch(
      'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    )
    if (!res.ok) {
      throw new Error('Failed to load image labels')
    }
    labels.value = await res.json()

    // Load ONNX model
    session = await ort.InferenceSession.create('/mobilenetv2-7.onnx', {
      executionProviders: ['wasm']
    })
    
    loading.value = false
  } catch (err) {
    console.error('❌ ONNX init error:', err)
    error.value = err.message || String(err)
    loading.value = false
  }
}

// Helper to wait for ort
async function waitForOrt(timeout = 5000) {
  const start = performance.now()
  while (!window.ort) {
    await new Promise(r => setTimeout(r, 100))
    if (performance.now() - start > timeout) {
      throw new Error('ONNX Runtime failed to load within timeout. Please refresh the page.')
    }
  }
}

async function handleFile(e) {
  const file = e.target.files[0]
  if (!file) return

  // Validate file type
  if (!file.type.startsWith('image/')) {
    error.value = 'Please select a valid image file'
    return
  }

  // Validate file size (max 10MB)
  if (file.size > 10 * 1024 * 1024) {
    error.value = 'Image file is too large. Please select an image under 10MB.'
    return
  }

  imageSrc.value = URL.createObjectURL(file)
  error.value = null
  prediction.value = ''

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

      // Find top prediction
      const data = output.data
      const topIdx = data.indexOf(Math.max(...data))
      const confidence = (data[topIdx] * 100).toFixed(1)
      const label = labels.value[topIdx] || 'Unknown'
      prediction.value = `${label} (${confidence}% confidence)`
      
      // Clean up tensor
      inputTensor.dispose()
      output.dispose()
    } catch (err) {
      console.error('❌ Prediction error:', err)
      error.value = 'Failed to process image. Please try again.'
    }
  }
  
  img.onerror = () => {
    error.value = 'Failed to load image. Please try a different file.'
  }
  
  img.src = imageSrc.value
}

function getTensorFromCanvas(canvasEl) {
  const ort = window.ort
  const ctx = canvasEl.getContext('2d', { willReadFrequently: true })
  const imageData = ctx.getImageData(0, 0, 224, 224)
  const input = new Float32Array(3 * 224 * 224)

  for (let i = 0; i < 224 * 224; i++) {
    input[i] = imageData.data[i * 4] / 255
    input[i + 224 * 224] = imageData.data[i * 4 + 1] / 255
    input[i + 2 * 224 * 224] = imageData.data[i * 4 + 2] / 255
  }

  return new ort.Tensor('float32', input, [1, 3, 224, 224])
}

onMounted(async () => {
  await initializeModel()
})
</script>

<style scoped>
.file-input-container {
  margin-bottom: 2rem;
}

.file-input-label {
  display: inline-block;
  padding: 1rem 2rem;
  background: #007bff;
  color: white;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.file-input-label:hover {
  background: #0056b3;
}

.file-input {
  display: none;
}

.image-preview {
  margin: 2rem 0;
}

.image-preview img {
  max-width: 300px;
  max-height: 300px;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.prediction-result {
  margin: 2rem 0;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
}

.prediction-text {
  font-size: 1.2rem;
  font-weight: bold;
  color: #28a745;
  margin: 0;
}

.error {
  margin: 2rem 0;
  padding: 1rem;
  background: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 8px;
  color: #721c24;
}

.retry-button {
  margin-top: 1rem;
  padding: 0.5rem 1rem;
  background: #dc3545;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.retry-button:hover {
  background: #c82333;
}

.loading {
  text-align: center;
  padding: 3rem;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #007bff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>

