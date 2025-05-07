<template>
  <div class="trainer-container">
    <canvas ref="canvas" class="bg-canvas" @mousemove="recordMouse" @mouseleave="pauseInteraction"
      @mouseenter="resumeInteraction"></canvas>

    <div class="overlay">
      <h2>▲ move mouse in box to train ▲</h2>
      <p>Current: ({{ mouseX }}, {{ mouseY }})</p>
      <p>Predicted: ({{ predictedX.toFixed(0) }}, {{ predictedY.toFixed(0) }})</p>
      <p>Learning: {{ training ? '▶' : '▷' }}</p>

    </div>

    
  </div>
  <!-- ──────────────── Explanation Block ──────────────── -->
    <div class="explanation">
    <h1>Behavior Modeling</h1>
    <h2>with TensorFlow / WebGPU</h2>
      <h3>🧠 What This Does</h3>
      <p>This demo uses TensorFlow.js to train a small neural network in real time using your mouse movement. It learns
        your motion pattern and predicts future pointer locations using only time as input.</p>

      <h3>🌐 Real‑World Applications</h3>
      <ul>
        <li>Predictive cursors and stylus smoothing</li>
        <li>Gesture learning for accessibility or automation</li>
        <li>Real-time inference of human behavior</li>
      </ul>

      <h3>🧩 Broader Uses for Time-Series + Haptics</h3>
      <ul>
        <li><strong>Touch Interfaces:</strong> Predict where a finger is going</li>
        <li><strong>Controllers:</strong> Smooth analog stick or gyro data</li>
        <li><strong>Health Devices:</strong> Learn motion habits via wearables</li>
      </ul>

      <h3>🔧 Tech Details</h3>
      <p>Powered by TensorFlow.js (WebGL backend). The model is trained on cursor positions over time and outputs a
        normalized position prediction, updated each frame. Only active while mouse is over the canvas.</p>
    </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import throttle from 'lodash-es/throttle'
import { initTF } from '../utils/tf-setup.js'

const mouseX = ref(0)
const mouseY = ref(0)
const predictedX = ref(0)
const predictedY = ref(0)
const canvas = ref(null)

let prevX = null
let prevY = null
let ctx, canvasWidth, canvasHeight
let model
let tf
let training = false
let allowTracking = false
let trainingData = []

const TRAIN_INTERVAL_MS = 200
const MOUSE_THROTTLE_MS = 50

const createModel = () => {
  model = tf.sequential()
  model.add(tf.layers.dense({ inputShape: [1], units: 8, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 4, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 2, activation: 'sigmoid' }))
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' })
}

const recordMouse = throttle((event) => {
  if (!allowTracking) return
  const rect = canvas.value.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  prevX = x
  prevY = y

  trainingData.push({
    t: (performance.now() % 10000) / 10000,
    x: x / canvasWidth,
    y: y / canvasHeight
  })

  mouseX.value = x
  mouseY.value = y
}, MOUSE_THROTTLE_MS)

const pauseInteraction = () => allowTracking = false
const resumeInteraction = () => allowTracking = true

const trainModel = async () => {
  if (training || trainingData.length < 10) return
  training = true

  if (trainingData.length > 1000) {
    trainingData.splice(0, trainingData.length - 1000)
  }

  const [xs, ys] = tf.tidy(() => {
    const times = trainingData.map(d => [d.t])
    const coords = trainingData.map(d => [d.x, d.y])
    return [tf.tensor2d(times), tf.tensor2d(coords)]
  })

  try {
    await model.fit(xs, ys, { epochs: 3, shuffle: true })
  } catch (err) {
    console.error('❌ Training error:', err)
  } finally {
    xs.dispose()
    ys.dispose()
    training = false
  }
}

const predict = async () => {
  const t = (performance.now() % 10000) / 10000
  const input = tf.tensor2d([[t]])
  const output = model.predict(input)
  const vals = await output.data()

  input.dispose()
  output.dispose()

  const x = vals[0] * canvasWidth
  const y = vals[1] * canvasHeight

  const clampedX = Math.min(Math.max(x, 0), canvasWidth)
  const clampedY = Math.min(Math.max(y, 0), canvasHeight)

  predictedX.value = clampedX
  predictedY.value = clampedY

  if (ctx) {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight)
    ctx.fillStyle = '#ff0000'
    ctx.beginPath()
    ctx.arc(clampedX, clampedY, 5, 0, Math.PI * 2)
    ctx.fill()
  }
}

let trainIntervalId
let rafId

onMounted(async () => {
  tf = await initTF()
  await nextTick()

  const cvs = canvas.value
  canvasWidth = 512
  canvasHeight = 512
  cvs.width = canvasWidth * devicePixelRatio
  cvs.height = canvasHeight * devicePixelRatio
  ctx = cvs.getContext('2d')
  ctx.scale(devicePixelRatio, devicePixelRatio)

  createModel()
  trainIntervalId = setInterval(trainModel, TRAIN_INTERVAL_MS)

  const loop = async () => {
    try {
      await predict()
    } catch (err) {
      console.error('Prediction error:', err)
    }
    rafId = requestAnimationFrame(loop)
  }

  loop()
})

onBeforeUnmount(() => {
  clearInterval(trainIntervalId)
  cancelAnimationFrame(rafId)
})
</script>


<style scoped>
.trainer-container {
  margin: 2rem auto;
  width: 512px;
  text-align: center;
}

.bg-canvas {
  display: block;
  border: 1px solid #ccc;
  margin: 1rem auto;
  width: 512px;
  height: 512px;
  background-color: #fff;
}

.overlay {
  color: #fff;
  font-family: monospace;
  margin-top: 1rem;
  text-align: left;
  width: fit-content;
  margin: 0 auto;
  margin-bottom: 5rem;
}

.overlay h2{
  color: #ff8800;
}

</style>
