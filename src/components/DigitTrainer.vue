<script setup lang="ts">
import * as tf from '@tensorflow/tfjs';
import { ref, onMounted, onBeforeUnmount } from 'vue';
import { initTF, getCurrentBackend, isWebGPUAvailable, isWebGLAvailable } from '../utils/tf-setup.js';
import DemoWrapper from './DemoWrapper.vue';

// Backend and model state
const currentBackend = ref('Initializing...');
const isWebGPU = ref(false);
const isWebGL = ref(false);
const status = ref('Initializing...');

// Canvas and drawing state
const size = 28;
const drawCanvas = ref<HTMLCanvasElement | null>(null);
const drawCtx = ref<CanvasRenderingContext2D | null>(null);
const drawing = ref(false);
const brush = 3;

// Model and data state
let model: tf.LayersModel | null = null;
let dsXs: tf.Tensor4D | null = null;
let dsYs: tf.Tensor2D | null = null;

// UI state
const selectedLabel = ref(0);
const predictText = ref('—');
const trainLog = ref('');

// Drawing functions
function clearCanvas() {
  if (!drawCtx.value || !drawCanvas.value) return;
  drawCtx.value.fillStyle = '#000';
  drawCtx.value.fillRect(0, 0, size, size);
}

function startDraw(e: MouseEvent) { 
  drawing.value = true; 
  paint(e); 
}

function endDraw() { 
  drawing.value = false; 
}

function paint(e: MouseEvent) {
  if (!drawing.value || !drawCtx.value || !drawCanvas.value) return;
  const r = drawCanvas.value.getBoundingClientRect();
  const x = (e.clientX - r.left) * (drawCanvas.value.width / r.width);
  const y = (e.clientY - r.top) * (drawCanvas.value.height / r.height);
  drawCtx.value.fillStyle = '#fff';
  drawCtx.value.beginPath();
  drawCtx.value.arc(x, y, brush, 0, Math.PI * 2);
  drawCtx.value.fill();
}

// Convert canvas to tensor
function imageTensorFromCanvas(canvas: HTMLCanvasElement): tf.Tensor4D {
  return tf.browser.fromPixels(canvas, 1).expandDims(0);
}

// Add sample to training data
async function addSample() {
  if (!drawCanvas.value) return;
  const x = imageTensorFromCanvas(drawCanvas.value);   // [1,28,28,1]
  const y = tf.oneHot(tf.tensor1d([selectedLabel.value], 'int32'), 10).toFloat(); // [1,10]
  
  if (dsXs && dsYs) {
    dsXs = tf.tidy(() => tf.concat([dsXs!, x], 0)) as tf.Tensor4D;
    dsYs = tf.tidy(() => tf.concat([dsYs!, y], 0)) as tf.Tensor2D;
    if (dsXs !== x) x.dispose();
    if (dsYs !== y) y.dispose();
  } else {
    dsXs = x;
    dsYs = y;
  }
  
  trainLog.value = `Added sample for digit ${selectedLabel.value}. Total samples: ${dsXs?.shape[0] || 0}`;
}

// Make prediction
async function predict() {
  if (!model || !drawCanvas.value) return;
  const x = imageTensorFromCanvas(drawCanvas.value);
  const p = model.predict(x) as tf.Tensor2D;
  const idx = (await p.argMax(1).data())[0];
  predictText.value = String(idx);
  p.dispose(); 
  x.dispose();
}

// Generate synthetic training data
async function seedSynthetic() {
  const { xs, ys } = makeSyntheticDigits(50); // 50 total samples
  
  if (dsXs && dsYs) {
    dsXs = tf.tidy(() => tf.concat([dsXs!, xs], 0)) as tf.Tensor4D;
    dsYs = tf.tidy(() => tf.concat([dsYs!, ys], 0)) as tf.Tensor2D;
    if (dsXs !== xs) xs.dispose(); 
    if (dsYs !== ys) ys.dispose();
  } else {
    dsXs = xs;
    dsYs = ys;
  }
  
  trainLog.value = `Added 50 synthetic samples. Total samples: ${dsXs?.shape[0] || 0}`;
}

// Create synthetic digit-like data
function makeSyntheticDigits(count: number) {
  const xs = tf.randomNormal([count, 28, 28, 1]);
  const ys = tf.randomUniform([count], 0, 10, 'int32');
  return { 
    xs, 
    ys: tf.oneHot(ys, 10)
  };
}

// Create CNN model
function makeDigitCnn(inputShape: number[], numClasses: number): tf.LayersModel {
  const model = tf.sequential();
  
  // Convolutional layers
  model.add(tf.layers.conv2d({
    inputShape,
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  
  // Dense layers
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
  
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

// Train the model
async function train() {
  if (!dsXs || !dsYs) { 
    alert('Add samples (or seed synthetic) first'); 
    return; 
  }
  
  trainLog.value = 'Training...';
  
  if (model) {
    model.dispose();
  }
  
  model = makeDigitCnn([28, 28, 1], 10);

  const history = await model.fit(dsXs, dsYs, {
    epochs: 8,
    batchSize: 128,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (e, logs) => {
        trainLog.value = `epoch ${e+1} | loss ${logs?.loss?.toFixed(3)} | acc ${(logs?.acc as number)?.toFixed?.(3)} | vAcc ${(logs?.val_acc as number)?.toFixed?.(3)}`;
      }
    }
  });
  
  const finalAcc = history.history.acc?.slice(-1)[0] as number;
  trainLog.value = `Done. acc ${finalAcc?.toFixed?.(3) || 'N/A'}`;
}

// Initialize demo
onMounted(async () => {
  try {
    await initTF();
    const backend = getCurrentBackend();
    currentBackend.value = backend;
    isWebGPU.value = await isWebGPUAvailable();
    isWebGL.value = await isWebGLAvailable();
    status.value = 'Ready';

    // Initialize canvas
    if (drawCanvas.value) {
      const c = drawCanvas.value;
      c.width = size; 
      c.height = size;
      drawCtx.value = c.getContext('2d')!;
      clearCanvas();
    }

    // Seed synthetic data for immediate training
    await seedSynthetic();
  } catch (error) {
    console.error('Initialization error:', error);
    currentBackend.value = 'Failed to initialize';
    status.value = 'Failed to initialize';
  }
});

// Cleanup
onBeforeUnmount(() => {
  if (model) {
    model.dispose();
  }
  if (dsXs) {
    dsXs.dispose();
  }
  if (dsYs) {
    dsYs.dispose();
  }
});
</script>

<template>
  <DemoWrapper>
    <template #tool>
      <div class="digit-trainer">
        <div class="status-bar">
          <p><strong>Status:</strong> {{ status }} | <strong>Backend:</strong> {{ currentBackend }}</p>
        </div>
        
        <div class="main-content">
          <div class="drawing-section">
            <h3>✏️ Draw a Digit</h3>
            <canvas 
              ref="drawCanvas" 
              class="draw-canvas"
              @mousedown="startDraw" 
              @mousemove="paint" 
              @mouseup="endDraw" 
              @mouseleave="endDraw"
              title="Draw a digit (0-9) here"
            ></canvas>
            
            <div class="drawing-controls">
              <button @click="clearCanvas" class="control-btn secondary">
                🗑️ Clear Canvas
              </button>
              
              <div class="label-selector">
                <label for="label-select">Label:</label>
                <select id="label-select" v-model="selectedLabel" class="label-select">
                  <option v-for="i in 10" :key="i-1" :value="i-1">{{ i-1 }}</option>
                </select>
              </div>
              
              <button @click="addSample" class="control-btn primary">
                ➕ Add Sample
              </button>
            </div>
          </div>
          
          <div class="training-section">
            <h3>🧠 Training & Prediction</h3>
            
            <div class="training-controls">
              <button @click="seedSynthetic" class="control-btn secondary">
                🚀 Seed Synthetic (50 samples)
              </button>
              
              <button @click="train" class="control-btn primary" :disabled="!dsXs || !dsYs">
                🎯 Train Model
              </button>
              
              <button @click="predict" class="control-btn success" :disabled="!model">
                🔮 Predict
              </button>
            </div>
            
            <div class="prediction-display">
              <strong>Prediction:</strong> 
              <span class="prediction-result">{{ predictText }}</span>
            </div>
            
            <div class="training-log">
              <strong>Training Log:</strong>
              <div class="log-content">{{ trainLog }}</div>
            </div>
          </div>
        </div>
      </div>
    </template>

    <template #explanation>
      <div class="explanation-content">
        <h2>Live Digit Trainer</h2>
        
        <h3>What This Demo Shows</h3>
        <p>
          This demo demonstrates <strong>interactive machine learning</strong> where you can draw your own digits, 
          train a model on them, and see real-time predictions. It's like teaching a computer to recognize 
          your handwriting!
        </p>
        
        <h3>How It Works</h3>
        <ol>
          <li><strong>Draw:</strong> Use your mouse to draw digits (0-9) on the canvas</li>
          <li><strong>Label:</strong> Select the correct label for what you drew</li>
          <li><strong>Add Sample:</strong> Add your drawing to the training dataset</li>
          <li><strong>Train:</strong> Train a neural network on your custom data</li>
          <li><strong>Predict:</strong> Test the model with new drawings</li>
        </ol>
        
        <h3>Key Features</h3>
        <ul>
          <li><strong>Interactive Drawing:</strong> Real-time digit drawing with mouse</li>
          <li><strong>Custom Training Data:</strong> Build your own dataset</li>
          <li><strong>Live Training:</strong> Watch the model learn in real-time</li>
          <li><strong>Instant Prediction:</strong> Test your trained model immediately</li>
          <li><strong>Synthetic Data:</strong> Quick-start with pre-generated samples</li>
        </ul>
        
        <h3>Real-World Applications</h3>
        <ul>
          <li><strong>Handwriting Recognition:</strong> OCR for forms, notes, and documents</li>
          <li><strong>Personal AI Training:</strong> Customize models to your writing style</li>
          <li><strong>Educational Tools:</strong> Teach AI concepts through hands-on experience</li>
          <li><strong>Prototyping:</strong> Quickly test ML ideas with custom data</li>
          <li><strong>Data Collection:</strong> Build specialized datasets for specific use cases</li>
        </ul>
        
        <h3>Technical Details</h3>
        <ul>
          <li><strong>Model Architecture:</strong> CNN with 32→64 filters, dropout, and 10-class classification</li>
          <li><strong>Input Format:</strong> 28×28 pixel grayscale images (MNIST standard)</li>
          <li><strong>Training:</strong> 8 epochs with validation split and real-time progress</li>
          <li><strong>Backend:</strong> Automatic fallback: WebGPU → WebGL → CPU</li>
          <li><strong>Memory Management:</strong> Efficient tensor handling and cleanup</li>
        </ul>
        
        <h3>Why This Matters</h3>
        <p>
          This demo shows how machine learning can be <strong>personalized and interactive</strong>. 
          Instead of using pre-trained models on generic data, you're creating a model that learns 
          from your specific examples. This approach is fundamental to many modern AI applications 
          where customization and user interaction are key.
        </p>
        
        <div class="note">
          <strong>Pro Tip:</strong> Start by drawing clear, centered digits. The model learns best from 
          consistent examples. Try drawing the same digit multiple times in slightly different ways to 
          improve recognition accuracy!
        </div>
      </div>
    </template>
  </DemoWrapper>
</template>

<style scoped>
.digit-trainer {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  width: 100%;
}

.status-bar {
  background: #e3f2fd;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #bbdefb;
  text-align: center;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  color: #1565c0;
}

.main-content {
  display: flex;
  gap: 2rem;
  flex-wrap: wrap;
  align-items: flex-start;
}

.drawing-section,
.training-section {
  flex: 1;
  min-width: 300px;
  background: #ffffff;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  border: 1px solid #e1e5e9;
}

.drawing-section h3,
.training-section h3 {
  margin: 0 0 1rem 0;
  color: #2c3e50;
  font-size: 1.3rem;
}

.draw-canvas {
  display: block;
  border: 2px solid #dee2e6;
  border-radius: 8px;
  margin: 0 auto 1rem auto;
  image-rendering: pixelated;
  cursor: crosshair;
}

.drawing-controls {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  align-items: center;
}

.label-selector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.label-selector label {
  font-weight: 600;
  color: #495057;
}

.label-select {
  padding: 0.5rem;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  background: #ffffff;
  font-size: 1rem;
  color: #495057;
  cursor: pointer;
}

.training-controls {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.prediction-display {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  margin-bottom: 1rem;
  text-align: center;
}

.prediction-result {
  font-size: 2rem;
  font-weight: bold;
  color: #28a745;
  margin-left: 0.5rem;
}

.training-log {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.log-content {
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
  color: #495057;
  margin-top: 0.5rem;
  white-space: pre-wrap;
  line-height: 1.4;
}

.control-btn {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  width: 100%;
}

.control-btn.primary {
  background: #007bff;
  color: white;
}

.control-btn.primary:hover:not(:disabled) {
  background: #0056b3;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
}

.control-btn.secondary {
  background: #6c757d;
  color: white;
}

.control-btn.secondary:hover:not(:disabled) {
  background: #545b62;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(108, 117, 125, 0.3);
}

.control-btn.success {
  background: #28a745;
  color: white;
}

.control-btn.success:hover:not(:disabled) {
  background: #1e7e34;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
}

.control-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

/* Responsive design */
@media (max-width: 768px) {
  .main-content {
    flex-direction: column;
  }
  
  .drawing-section,
  .training-section {
    min-width: auto;
  }
  
  .draw-canvas {
    width: 200px;
    height: 200px;
  }
}

.explanation-content h2 {
  color: #2c3e50;
  margin-bottom: 1.5rem;
  font-size: 1.8rem;
}

.explanation-content h3 {
  color: #34495e;
  margin: 1.5rem 0 0.75rem 0;
  font-size: 1.2rem;
}

.explanation-content p {
  line-height: 1.6;
  margin-bottom: 1rem;
  color: #555;
}

.explanation-content ul,
.explanation-content ol {
  margin: 1rem 0;
  padding-left: 1.5rem;
}

.explanation-content li {
  margin-bottom: 0.5rem;
  line-height: 1.5;
  color: #555;
}

.note {
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 6px;
  padding: 1rem;
  margin-top: 1.5rem;
  color: #856404;
}

.note strong {
  color: #6c5ce7;
}
</style>
