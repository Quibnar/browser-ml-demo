<template>
  <DemoWrapper>
    <!-- TOOL SLOT -->
    <template #tool>
      <div class="trainer-container">
        <canvas ref="canvas" class="bg-canvas" @mousemove="recordMouse" @mouseleave="pauseInteraction"
          @mouseenter="resumeInteraction"></canvas>

        <div class="overlay">
          <h2>▲ move mouse in box to train ▲</h2>
          <p>Current: ({{ mouseX }}, {{ mouseY }})</p>
          <p>Predicted: ({{ predictedX.toFixed(0) }}, {{ predictedY.toFixed(0) }})</p>
          <p>Learning: {{ training ? '▶' : '▷' }}</p>
          <p>Backend: {{ currentBackend }} {{ isWebGPU ? '🚀' : isWebGL ? '⚡' : '🐌' }}</p>
          <p>Training Speed: {{ trainingSpeed }} samples/sec</p>
          <p>Model Loss: {{ modelLoss.toFixed(4) }}</p>
          <p>Status: {{ allowTracking ? '🟢 Training' : '🟡 Predicting' }}</p>
          <p>Stable Patterns: {{ stablePatternCount }}</p>
          <p>Exit Gestures Filtered: {{ exitGesturesFiltered }}</p>
          <p>Heat Map Intensity: {{ heatMapIntensity }}</p>
          <p>Flow Field Strength: {{ flowFieldStrength }}</p>
          <p>Repeating Patterns: {{ repeatingPatternCount }}</p>
        </div>

        <div class="controls">
          <button @click="resetModel" class="control-btn">🔄 Reset Model</button>
          <button @click="toggleTraining" class="control-btn">{{ training ? '⏸️ Pause' : '▶️ Resume' }}</button>
          <button @click="exportModel" class="control-btn">💾 Export</button>
          <button @click="toggleHeatMap" class="control-btn">{{ showHeatMap ? '🌡️ Hide Heat' : '🌡️ Show Heat' }}</button>
        </div>
      </div>
    </template>

    <!-- EXPLANATION SLOT -->
    <template #explanation>
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
      
      <div class="note">
        <p><strong>Note:</strong> On Windows, you may see a WebGPU warning about "powerPreference option ignored" - this is normal and doesn't affect performance. WebGPU will still work optimally!</p>
      </div>
    </template>
  </DemoWrapper>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick, computed } from 'vue'
import throttle from 'lodash-es/throttle'
import { initTF, getCurrentBackend, isWebGPUAvailable, isWebGLAvailable } from '../utils/tf-setup.js'
import DemoWrapper from './DemoWrapper.vue'

const mouseX = ref(0)
const mouseY = ref(0)
const predictedX = ref(0)
const predictedY = ref(0)
const canvas = ref(null)
const currentBackend = ref('Unknown')
const isWebGPU = ref(false)
const isWebGL = ref(false)
const trainingSpeed = ref(0)
const modelLoss = ref(0)
const showHeatMap = ref(true)

let prevX = null
let prevY = null
let ctx, canvasWidth, canvasHeight
let model
let tf
let training = false
let allowTracking = false
let trainingData = []
let isActive = false
let trainingStartTime = 0
let sampleCount = 0

// Performance tuning constants
const TRAIN_INTERVAL_MS = 100 // Reduced from 200ms for faster learning
const MOUSE_THROTTLE_MS = 16 // 60fps sampling (reduced from 50ms)
const BATCH_SIZE = 32 // Process multiple samples at once
const MAX_TRAINING_DATA = 2000 // Increased from 1000 for better learning
const LEARNING_RATE = 0.001
const ADAPTIVE_LEARNING = true

// Pattern preservation constants
const EXIT_GESTURE_THRESHOLD = 0.8 // Distance threshold for exit detection
const PATTERN_FREQUENCY_WEIGHT = 0.7 // Weight for frequently occurring patterns
const RECENT_SAMPLE_WEIGHT = 0.3 // Weight for recent samples
const MIN_PATTERN_SAMPLES = 50 // Minimum samples before considering a pattern stable
const EXIT_GESTURE_FILTER = true // Enable exit gesture filtering

// Heat map and pattern analysis constants
const HEATMAP_RESOLUTION = 128 // Increased from 32 to 128x128 for higher fidelity
const HEATMAP_DECAY_RATE = 0.995 // Slower decay (99.5% retention) for longer memory
const HEATMAP_MEMORY_LAYERS = 10 // Store 10 temporal layers for history
const PATTERN_MEMORY_SIZE = 2000 // Increased from 1000 for longer story
const SPATIAL_CLUSTER_RADIUS = 0.03 // Reduced from 0.05 for finer clustering
const TEMPORAL_WINDOW_SIZE = 200 // Increased from 100 for longer pattern analysis
const REPEATING_PATTERN_THRESHOLD = 0.6 // Similarity threshold for pattern matching

// Flow field constants
const FLOW_FIELD_RESOLUTION = 64 // 64x64 grid for flow field (balance of detail vs performance)
const FLOW_FIELD_DECAY_RATE = 0.99 // Flow field decays slower than heat map
const FLOW_FIELD_SMOOTHING = 0.3 // Smoothing factor for flow field updates
const MIN_FLOW_MAGNITUDE = 0.01 // Minimum movement to record in flow field

// Memory pool for tensor reuse
let tensorPool = {
  input: null,
  output: null,
  times: null,
  coords: null
}

// Enhanced pattern analysis and storage
let patternAnalysis = {
  frequentPatterns: [], // Store frequently occurring movement patterns
  patternWeights: new Map(), // Weight each pattern by frequency
  exitGestures: [], // Track potential exit gestures
  stablePatterns: new Set(), // Patterns that are stable and shouldn't be overridden
  
  // Enhanced heat map with temporal layers
  heatMap: new Array(HEATMAP_RESOLUTION * HEATMAP_RESOLUTION).fill(0), // Current heat map
  heatMapHistory: [], // Store last 10 heat map layers for temporal analysis
  heatMapPeaks: new Map(), // Track peak heat values for each cell
  
  // Flow field for directional patterns
  flowField: {
    directions: new Array(FLOW_FIELD_RESOLUTION * FLOW_FIELD_RESOLUTION).fill(0), // Direction angles
    magnitudes: new Array(FLOW_FIELD_RESOLUTION * FLOW_FIELD_RESOLUTION).fill(0), // Movement magnitudes
    velocities: new Array(FLOW_FIELD_RESOLUTION * FLOW_FIELD_RESOLUTION).fill(0), // Velocity history
    timestamps: new Array(FLOW_FIELD_RESOLUTION * FLOW_FIELD_RESOLUTION).fill(0) // Last update time
  },
  
  temporalPatterns: [], // Store temporal sequences for pattern matching
  spatialClusters: new Map(), // Group similar spatial positions
  repeatingPatterns: new Map() // Track patterns that repeat over time
}

// Computed properties for UI
const stablePatternCount = computed(() => patternAnalysis.stablePatterns.size)
const exitGesturesFiltered = computed(() => patternAnalysis.exitGestures.length)
const heatMapIntensity = computed(() => {
  const maxHeat = Math.max(...patternAnalysis.heatMap)
  return maxHeat > 0 ? (maxHeat / 100).toFixed(2) : '0.00'
})
const repeatingPatternCount = computed(() => patternAnalysis.repeatingPatterns.size)
const flowFieldStrength = computed(() => {
  const avgMagnitude = patternAnalysis.flowField.magnitudes.reduce((a, b) => a + b, 0) / patternAnalysis.flowField.magnitudes.length
  return avgMagnitude > 0 ? (avgMagnitude * 100).toFixed(1) : '0.0'
})

const createModel = () => {
  // Create a more sophisticated model for better learning
  model = tf.sequential()
  
  // Input layer with normalization
  model.add(tf.layers.dense({ 
    inputShape: [1], 
    units: 16, // Increased from 8
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }))
  
  // Hidden layers with dropout for regularization
  model.add(tf.layers.dropout({ rate: 0.1 }))
  model.add(tf.layers.dense({ 
    units: 32, // Increased from 4
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }))
  
  model.add(tf.layers.dropout({ rate: 0.1 }))
  model.add(tf.layers.dense({ 
    units: 16, 
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }))
  
  // Output layer
  model.add(tf.layers.dense({ 
    units: 2, 
    activation: 'sigmoid',
    kernelInitializer: 'glorotNormal'
  }))
  
  // Use adaptive learning rate optimizer
  const optimizer = ADAPTIVE_LEARNING 
    ? tf.train.adamax(LEARNING_RATE)
    : tf.train.adam(LEARNING_RATE)
  
  model.compile({ 
    optimizer, 
    loss: 'meanSquaredError',
    metrics: ['accuracy']
  })
}

// Enhanced heat map with temporal layers
const updateHeatMap = (x, y, velocity = 0) => {
  // Convert canvas coordinates to heat map grid
  const gridX = Math.floor((x / canvasWidth) * HEATMAP_RESOLUTION)
  const gridY = Math.floor((y / canvasHeight) * HEATMAP_RESOLUTION)
  
  // Add heat to current position and surrounding cells with velocity influence
  const heatRadius = Math.min(3, Math.floor(velocity / 50) + 1) // Larger radius for faster movement
  
  for (let dx = -heatRadius; dx <= heatRadius; dx++) {
    for (let dy = -heatRadius; dy <= heatRadius; dy++) {
      const nx = gridX + dx
      const ny = gridY + dy
      
      if (nx >= 0 && nx < HEATMAP_RESOLUTION && ny >= 0 && ny < HEATMAP_RESOLUTION) {
        const index = ny * HEATMAP_RESOLUTION + nx
        const distance = Math.sqrt(dx * dx + dy * dy)
        const heat = Math.max(0, (1 - distance / heatRadius) * (1 + velocity / 100)) // Velocity boosts heat
        
        patternAnalysis.heatMap[index] += heat
        
        // Track peak heat values
        const currentPeak = patternAnalysis.heatMapPeaks.get(index) || 0
        if (heat > currentPeak) {
          patternAnalysis.heatMapPeaks.set(index, heat)
        }
      }
    }
  }
  
  // Apply slower decay to all cells for longer memory
  patternAnalysis.heatMap = patternAnalysis.heatMap.map(heat => heat * HEATMAP_DECAY_RATE)
  
  // Store heat map history every 100 frames
  if (patternAnalysis.heatMapHistory.length === 0 || 
      patternAnalysis.heatMapHistory.length < HEATMAP_MEMORY_LAYERS) {
    patternAnalysis.heatMapHistory.push([...patternAnalysis.heatMap])
  } else {
    patternAnalysis.heatMapHistory.shift()
    patternAnalysis.heatMapHistory.push([...patternAnalysis.heatMap])
  }
}

// Update flow field with directional information
const updateFlowField = (x, y, dx, dy, timestamp) => {
  // Convert canvas coordinates to flow field grid
  const gridX = Math.floor((x / canvasWidth) * FLOW_FIELD_RESOLUTION)
  const gridY = Math.floor((y / canvasHeight) * FLOW_FIELD_RESOLUTION)
  
  if (gridX < 0 || gridX >= FLOW_FIELD_RESOLUTION || gridY < 0 || gridY >= FLOW_FIELD_RESOLUTION) return
  
  const index = gridY * FLOW_FIELD_RESOLUTION + gridX
  const magnitude = Math.sqrt(dx * dx + dy * dy)
  
  // Only update if movement is significant
  if (magnitude < MIN_FLOW_MAGNITUDE) return
  
  // Calculate direction angle
  const angle = Math.atan2(dy, dx)
  
  // Smooth update with existing flow field
  const existingAngle = patternAnalysis.flowField.directions[index]
  const existingMagnitude = patternAnalysis.flowField.magnitudes[index]
  
  // Weighted average for smooth transitions
  const newAngle = existingMagnitude > 0 
    ? Math.atan2(
        Math.sin(existingAngle) * (1 - FLOW_FIELD_SMOOTHING) + Math.sin(angle) * FLOW_FIELD_SMOOTHING,
        Math.cos(existingAngle) * (1 - FLOW_FIELD_SMOOTHING) + Math.cos(angle) * FLOW_FIELD_SMOOTHING
      )
    : angle
  
  const newMagnitude = Math.max(existingMagnitude, magnitude)
  
  // Update flow field
  patternAnalysis.flowField.directions[index] = newAngle
  patternAnalysis.flowField.magnitudes[index] = newMagnitude
  patternAnalysis.flowField.velocities[index] = magnitude
  patternAnalysis.flowField.timestamps[index] = timestamp
  
  // Apply decay to flow field
  patternAnalysis.flowField.magnitudes[index] *= FLOW_FIELD_DECAY_RATE
  patternAnalysis.flowField.velocities[index] *= FLOW_FIELD_DECAY_RATE
}

// Enhanced heat map drawing with temporal layers
const drawHeatMap = () => {
  if (!ctx || !showHeatMap.value) return
  
  const cellWidth = canvasWidth / HEATMAP_RESOLUTION
  const cellHeight = canvasHeight / HEATMAP_RESOLUTION
  
  // Find max heat for normalization
  const maxHeat = Math.max(...patternAnalysis.heatMap)
  if (maxHeat === 0) return
  
  // Draw heat map as semi-transparent overlay with temporal depth
  for (let y = 0; y < HEATMAP_RESOLUTION; y++) {
    for (let x = 0; x < HEATMAP_RESOLUTION; x++) {
      const index = y * HEATMAP_RESOLUTION + x
      const currentHeat = patternAnalysis.heatMap[index] / maxHeat
      
      // Calculate temporal depth (how long heat has been present)
      let temporalDepth = 0
      for (let layer = 0; layer < patternAnalysis.heatMapHistory.length; layer++) {
        const historicalHeat = patternAnalysis.heatMapHistory[layer][index] / maxHeat
        temporalDepth += historicalHeat * (layer + 1) / patternAnalysis.heatMapHistory.length
      }
      
      // Combine current heat with temporal depth
      const combinedHeat = Math.max(currentHeat, temporalDepth * 0.5)
      
      if (combinedHeat > 0.05) { // Only draw cells with significant heat
        const alpha = Math.min(0.4, combinedHeat * 0.4) // Max 40% opacity
        const intensity = Math.min(1.0, combinedHeat * 2.0) // Boost intensity for visibility
        
        // Color based on heat intensity and temporal depth
        let r, g, b
        if (intensity < 0.5) {
          // Cool colors for low heat
          r = 100 + Math.floor(intensity * 155)
          g = 100 + Math.floor(intensity * 155)
          b = 255
        } else {
          // Warm colors for high heat
          r = 255
          g = 100 + Math.floor((intensity - 0.5) * 155)
          b = 100
        }
        
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`
        ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight)
      }
    }
  }
}

// Draw flow field arrows
const drawFlowField = () => {
  if (!ctx || !showHeatMap.value) return
  
  const cellWidth = canvasWidth / FLOW_FIELD_RESOLUTION
  const cellHeight = canvasHeight / FLOW_FIELD_RESOLUTION
  
  ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)'
  ctx.lineWidth = 1
  
  for (let y = 0; y < FLOW_FIELD_RESOLUTION; y++) {
    for (let x = 0; x < FLOW_FIELD_RESOLUTION; x++) {
      const index = y * FLOW_FIELD_RESOLUTION + x
      const magnitude = patternAnalysis.flowField.magnitudes[index]
      const direction = patternAnalysis.flowField.directions[index]
      
      if (magnitude > 0.02) { // Only draw significant flow
        const centerX = (x + 0.5) * cellWidth
        const centerY = (y + 0.5) * cellHeight
        
        // Calculate arrow length based on magnitude
        const arrowLength = Math.min(cellWidth * 0.8, magnitude * 100)
        
        // Draw arrow
        const endX = centerX + Math.cos(direction) * arrowLength
        const endY = centerY + Math.sin(direction) * arrowLength
        
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(endX, endY)
        ctx.stroke()
        
        // Draw arrowhead
        const arrowheadLength = arrowLength * 0.3
        const arrowheadAngle = Math.PI / 6
        
        ctx.beginPath()
        ctx.moveTo(endX, endY)
        ctx.lineTo(
          endX - arrowheadLength * Math.cos(direction - arrowheadAngle),
          endY - arrowheadLength * Math.sin(direction - arrowheadAngle)
        )
        ctx.moveTo(endX, endY)
        ctx.lineTo(
          endX - arrowheadLength * Math.cos(direction + arrowheadAngle),
          endY - arrowheadLength * Math.sin(direction + arrowheadAngle)
        )
        ctx.stroke()
      }
    }
  }
}

// Enhanced pattern analysis with flow field integration
const analyzeMovementPattern = (newData) => {
  if (trainingData.length < 10) return true
  
  const recentSamples = trainingData.slice(-10)
  
  // Calculate movement characteristics
  const distances = []
  const velocities = []
  const directions = []
  
  for (let i = 1; i < recentSamples.length; i++) {
    const prev = recentSamples[i - 1]
    const curr = recentSamples[i]
    
    const dx = (curr.x - prev.x) * canvasWidth
    const dy = (curr.y - prev.y) * canvasHeight
    const distance = Math.sqrt(dx * dx + dy * dy)
    const velocity = distance / (curr.timestamp - prev.timestamp)
    const direction = Math.atan2(dy, dx)
    
    distances.push(distance)
    velocities.push(velocity)
    directions.push(direction)
  }
  
  // Update heat map and flow field
  const avgVelocity = velocities.reduce((a, b) => a + b, 0) / velocities.length
  updateHeatMap(newData.x * canvasWidth, newData.y * canvasHeight, avgVelocity)
  
  // Update flow field with movement direction
  if (recentSamples.length > 1) {
    const lastSample = recentSamples[recentSamples.length - 1]
    const dx = (newData.x - lastSample.x) * canvasWidth
    const dy = (newData.y - lastSample.y) * canvasHeight
    updateFlowField(newData.x * canvasWidth, newData.y * canvasHeight, dx, dy, newData.timestamp)
  }
  
  // Detect potential exit gesture (sudden large movement)
  const avgDistance = distances.reduce((a, b) => a + b, 0) / distances.length
  
  const isExitGesture = avgDistance > EXIT_GESTURE_THRESHOLD * 100 || 
                       avgVelocity > 2.0 // High velocity threshold
  
  if (isExitGesture && EXIT_GESTURE_FILTER) {
    patternAnalysis.exitGestures.push({
      timestamp: Date.now(),
      samples: recentSamples,
      avgDistance,
      avgVelocity
    })
    
    // Don't add exit gesture samples to training data
    return false
  }
  
  // Enhanced pattern analysis with flow field integration
  const spatialKey = generateSpatialKey(recentSamples)
  const temporalKey = generateTemporalKey(recentSamples)
  const flowKey = generateFlowKey(recentSamples)
  
  // Update spatial clusters
  if (!patternAnalysis.spatialClusters.has(spatialKey)) {
    patternAnalysis.spatialClusters.set(spatialKey, [])
  }
  patternAnalysis.spatialClusters.get(spatialKey).push(newData)
  
  // Update temporal patterns
  if (patternAnalysis.temporalPatterns.length >= PATTERN_MEMORY_SIZE) {
    patternAnalysis.temporalPatterns.shift()
  }
  patternAnalysis.temporalPatterns.push(temporalKey)
  
  // Detect repeating patterns
  detectRepeatingPatterns(temporalKey)
  
  // Analyze pattern frequency with enhanced weighting including flow field
  const patternKey = generatePatternKey(recentSamples)
  const currentWeight = patternAnalysis.patternWeights.get(patternKey) || 0
  const spatialFrequency = getSpatialFrequency(spatialKey)
  const temporalFrequency = getTemporalFrequency(temporalKey)
  const flowConsistency = getFlowConsistency(flowKey)
  
  // Enhanced weight calculation considering spatial, temporal, and flow consistency
  const enhancedWeight = currentWeight + 1 + 
                        (spatialFrequency * 0.4) + 
                        (temporalFrequency * 0.3) + 
                        (flowConsistency * 0.3)
  
  patternAnalysis.patternWeights.set(patternKey, enhancedWeight)
  
  // Mark patterns as stable if they occur frequently
  if (enhancedWeight >= MIN_PATTERN_SAMPLES) {
    patternAnalysis.stablePatterns.add(patternKey)
  }
  
  return true
}

// Generate flow field key for directional pattern analysis
const generateFlowKey = (samples) => {
  if (samples.length < 3) return 'short'
  
  const flowPatterns = []
  for (let i = 1; i < samples.length; i++) {
    const prev = samples[i - 1]
    const curr = samples[i]
    
    const dx = curr.x - prev.x
    const dy = curr.y - prev.y
    const angle = Math.atan2(dy, dx)
    
    // Quantize direction into 8 sectors
    const sector = Math.round(angle / (Math.PI / 4)) % 8
    const magnitude = Math.sqrt(dx * dx + dy * dy)
    
    flowPatterns.push(`${sector}:${magnitude > 0.05 ? 'L' : 'S'}`)
  }
  
  return flowPatterns.join(',')
}

// Get flow consistency for a given flow key
const getFlowConsistency = (flowKey) => {
  if (flowKey === 'short') return 0
  
  const parts = flowKey.split(',')
  const directions = parts.map(p => parseInt(p.split(':')[0]))
  
  // Calculate consistency (how similar consecutive directions are)
  let consistency = 0
  for (let i = 1; i < directions.length; i++) {
    const diff = Math.abs(directions[i] - directions[i - 1])
    const normalizedDiff = Math.min(diff, 8 - diff) / 4 // Normalize to 0-1
    consistency += (1 - normalizedDiff)
  }
  
  return consistency / (directions.length - 1)
}

// Generate spatial key for clustering similar positions
const generateSpatialKey = (samples) => {
  if (samples.length < 3) return 'short'
  
  // Group samples into spatial clusters
  const positions = samples.map(s => ({ x: s.x, y: s.y }))
  const clusters = []
  
  for (const pos of positions) {
    let addedToCluster = false
    for (const cluster of clusters) {
      const centerX = cluster.reduce((sum, p) => sum + p.x, 0) / cluster.length
      const centerY = cluster.reduce((sum, p) => sum + p.y, 0) / cluster.length
      const distance = Math.sqrt((pos.x - centerX) ** 2 + (pos.y - centerY) ** 2)
      
      if (distance < SPATIAL_CLUSTER_RADIUS) {
        cluster.push(pos)
        addedToCluster = true
        break
      }
    }
    
    if (!addedToCluster) {
      clusters.push([pos])
    }
  }
  
  // Create spatial signature
  return clusters
    .filter(cluster => cluster.length > 1)
    .map(cluster => `${cluster.length}:${cluster.length > 2 ? 'D' : 'S'}`)
    .sort()
    .join(',')
}

// Generate temporal key for sequence analysis
const generateTemporalKey = (samples) => {
  if (samples.length < 5) return 'short'
  
  // Create temporal sequence signature
  const movements = []
  for (let i = 1; i < samples.length; i++) {
    const prev = samples[i - 1]
    const curr = samples[i]
    
    const dx = curr.x - prev.x
    const dy = curr.y - prev.y
    
    // Quantize movement into 16 directions for finer temporal analysis
    const angle = Math.atan2(dy, dx)
    const direction = Math.round(angle / (Math.PI / 8)) % 16
    const magnitude = Math.sqrt(dx * dx + dy * dy)
    
    movements.push(`${direction}:${magnitude > 0.05 ? 'L' : magnitude > 0.02 ? 'M' : 'S'}`)
  }
  
  return movements.join(',')
}

// Generate a pattern key for frequency analysis
const generatePatternKey = (samples) => {
  if (samples.length < 3) return 'short'
  
  // Create a simplified pattern signature
  const movements = []
  for (let i = 1; i < samples.length; i++) {
    const prev = samples[i - 1]
    const curr = samples[i]
    
    const dx = curr.x - prev.x
    const dy = curr.y - prev.y
    
    // Quantize movement into 8 directions
    const angle = Math.atan2(dy, dx)
    const direction = Math.round(angle / (Math.PI / 4)) % 8
    const magnitude = Math.sqrt(dx * dx + dy * dy)
    
    movements.push(`${direction}:${magnitude > 0.1 ? 'L' : 'S'}`)
  }
  
  return movements.join(',')
}

// Detect repeating patterns in temporal sequences
const detectRepeatingPatterns = (currentPattern) => {
  if (patternAnalysis.temporalPatterns.length < TEMPORAL_WINDOW_SIZE) return
  
  const recentPatterns = patternAnalysis.temporalPatterns.slice(-TEMPORAL_WINDOW_SIZE)
  const patternCounts = new Map()
  
  // Count pattern occurrences
  for (const pattern of recentPatterns) {
    patternCounts.set(pattern, (patternCounts.get(pattern) || 0) + 1)
  }
  
  // Find patterns that repeat significantly
  for (const [pattern, count] of patternCounts) {
    if (count >= 3 && pattern !== 'short') { // Pattern appears at least 3 times
      const similarity = calculatePatternSimilarity(currentPattern, pattern)
      if (similarity > REPEATING_PATTERN_THRESHOLD) {
        patternAnalysis.repeatingPatterns.set(pattern, {
          count,
          lastSeen: Date.now(),
          frequency: count / TEMPORAL_WINDOW_SIZE
        })
      }
    }
  }
}

// Calculate similarity between two patterns
const calculatePatternSimilarity = (pattern1, pattern2) => {
  if (pattern1 === pattern2) return 1.0
  
  const parts1 = pattern1.split(',')
  const parts2 = pattern2.split(',')
  
  if (parts1.length !== parts2.length) return 0.0
  
  let matches = 0
  for (let i = 0; i < parts1.length; i++) {
    if (parts1[i] === parts2[i]) matches++
  }
  
  return matches / parts1.length
}

// Get spatial frequency for a given spatial key
const getSpatialFrequency = (spatialKey) => {
  const cluster = patternAnalysis.spatialClusters.get(spatialKey)
  return cluster ? cluster.length : 0
}

// Get temporal frequency for a given temporal key
const getTemporalFrequency = (temporalKey) => {
  const recentPatterns = patternAnalysis.temporalPatterns.slice(-TEMPORAL_WINDOW_SIZE)
  return recentPatterns.filter(p => p === temporalKey).length
}

const recordMouse = throttle((event) => {
  if (!allowTracking || !isActive) return
  const rect = canvas.value.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  prevX = x
  prevY = y

  // Add timestamp with higher precision
  const timestamp = performance.now()
  const newData = {
    t: (timestamp % 10000) / 10000,
    x: x / canvasWidth,
    y: y / canvasHeight,
    timestamp
  }
  
  // Analyze the movement pattern before adding to training data
  const shouldAdd = analyzeMovementPattern(newData)
  
  if (shouldAdd) {
    trainingData.push(newData)
    sampleCount++
  }

  mouseX.value = x
  mouseY.value = y
}, MOUSE_THROTTLE_MS)

const pauseInteraction = () => {
  allowTracking = false
  // Don't pause isActive - let the model continue predicting!
  // This is the key change - the model keeps running even when mouse leaves
}

const resumeInteraction = () => {
  allowTracking = true
  isActive = true
}

const resetModel = () => {
  if (model) {
    model.dispose()
  }
  trainingData = []
  sampleCount = 0
  modelLoss.value = 0
  // Reset pattern analysis
  patternAnalysis = {
    frequentPatterns: [],
    patternWeights: new Map(),
    exitGestures: [],
    stablePatterns: new Set(),
    heatMap: new Array(HEATMAP_RESOLUTION * HEATMAP_RESOLUTION).fill(0),
    heatMapHistory: [],
    heatMapPeaks: new Map(),
    flowField: {
      directions: new Array(FLOW_FIELD_RESOLUTION * FLOW_FIELD_RESOLUTION).fill(0),
      magnitudes: new Array(FLOW_FIELD_RESOLUTION * FLOW_FIELD_RESOLUTION).fill(0),
      velocities: new Array(FLOW_FIELD_RESOLUTION * FLOW_FIELD_RESOLUTION).fill(0),
      timestamps: new Array(FLOW_FIELD_RESOLUTION * FLOW_FIELD_RESOLUTION).fill(0)
    },
    temporalPatterns: [],
    spatialClusters: new Map(),
    repeatingPatterns: new Map()
  }
  createModel()
}

const toggleTraining = () => {
  isActive = !isActive
  if (isActive) {
    resumeInteraction()
  } else {
    pauseInteraction()
  }
}

const toggleHeatMap = () => {
  showHeatMap.value = !showHeatMap.value
}

const exportModel = async () => {
  if (model) {
    try {
      await model.save('downloads://mouse-predictor-model')
      console.log('✅ Model exported successfully')
    } catch (err) {
      console.error('❌ Export failed:', err)
    }
  }
}

// Enhanced sample selection with heat map and flow field influence
const selectTrainingSamples = () => {
  if (trainingData.length <= BATCH_SIZE) {
    return trainingData.map(d => ({ ...d, weight: 1 }))
  }
  
  const samples = []
  const recentSamples = trainingData.slice(-BATCH_SIZE / 3)
  const historicalSamples = trainingData.slice(0, -BATCH_SIZE / 3)
  
  // Add recent samples (weighted by recency)
  recentSamples.forEach((sample, index) => {
    const weight = RECENT_SAMPLE_WEIGHT * (index + 1) / recentSamples.length
    samples.push({ ...sample, weight })
  })
  
  // Add historical samples (weighted by pattern frequency, heat map, and flow field)
  const patternSamples = historicalSamples.filter(sample => {
    const patternKey = generatePatternKey([sample])
    return patternAnalysis.stablePatterns.has(patternKey)
  })
  
  // Select samples with high heat map values, pattern frequency, and flow consistency
  const enhancedSamples = patternSamples
    .map(sample => {
      const patternKey = generatePatternKey([sample])
      const frequency = patternAnalysis.patternWeights.get(patternKey) || 1
      
      // Calculate heat map influence
      const gridX = Math.floor(sample.x * HEATMAP_RESOLUTION)
      const gridY = Math.floor(sample.y * HEATMAP_RESOLUTION)
      const heatIndex = gridY * HEATMAP_RESOLUTION + gridX
      const heatValue = patternAnalysis.heatMap[heatIndex] || 0
      
      // Calculate flow field influence
      const flowGridX = Math.floor(sample.x * FLOW_FIELD_RESOLUTION)
      const flowGridY = Math.floor(sample.y * FLOW_FIELD_RESOLUTION)
      const flowIndex = flowGridY * FLOW_FIELD_RESOLUTION + flowGridX
      const flowValue = patternAnalysis.flowField.magnitudes[flowIndex] || 0
      
      // Enhanced weight calculation
      const baseWeight = PATTERN_FREQUENCY_WEIGHT * Math.min(frequency / MIN_PATTERN_SAMPLES, 1)
      const heatWeight = Math.min(heatValue / 10, 0.3) // Heat contributes up to 30% of weight
      const flowWeight = Math.min(flowValue / 2, 0.2) // Flow field contributes up to 20% of weight
      
      return { ...sample, weight: baseWeight + heatWeight + flowWeight }
    })
    .sort((a, b) => b.weight - a.weight)
    .slice(0, BATCH_SIZE * 2 / 3)
  
  samples.push(...enhancedSamples)
  return samples.slice(0, BATCH_SIZE)
}

const trainModel = async () => {
  if (training || trainingData.length < BATCH_SIZE || !isActive) return
  training = true
  trainingStartTime = performance.now()

  // Keep only recent data for faster training
  if (trainingData.length > MAX_TRAINING_DATA) {
    trainingData.splice(0, trainingData.length - MAX_TRAINING_DATA)
  }

  try {
    // Use smart sample selection for training
    const selectedSamples = selectTrainingSamples()
    
    // Use tensor pooling to reduce memory allocations
    const [xs, ys] = tf.tidy(() => {
      // Create weighted training data
      const times = selectedSamples.map(d => [d.t])
      const coords = selectedSamples.map(d => [d.x, d.y])
      
      return [tf.tensor2d(times), tf.tensor2d(coords)]
    })

    // Train with adaptive learning rate and sample weights
    const sampleWeights = selectedSamples.map(d => d.weight || 1)
    const sampleWeightsTensor = tf.tensor1d(sampleWeights)
    
    const history = await model.fit(xs, ys, { 
      epochs: 2, // Reduced from 3 for faster iteration
      batchSize: BATCH_SIZE,
      shuffle: true,
      verbose: 0,
      sampleWeights: sampleWeightsTensor
    })

    // Update metrics
    if (history.history.loss && history.history.loss.length > 0) {
      modelLoss.value = history.history.loss[history.history.loss.length - 1]
    }

    // Calculate training speed
    const trainingTime = performance.now() - trainingStartTime
    trainingSpeed.value = Math.round((BATCH_SIZE / trainingTime) * 1000)

    // Clean up tensors
    xs.dispose()
    ys.dispose()
    sampleWeightsTensor.dispose()

  } catch (err) {
    console.error('❌ Training error:', err)
  } finally {
    training = false
  }
}

const predict = async () => {
  if (!isActive || !model) return
  
  const t = (performance.now() % 10000) / 10000
  
  try {
    // Use tensor pooling for prediction
    const input = tf.tensor2d([[t]])
    const output = model.predict(input)
    const vals = await output.data()

    const x = vals[0] * canvasWidth
    const y = vals[1] * canvasHeight

    const clampedX = Math.min(Math.max(x, 0), canvasWidth)
    const clampedY = Math.min(Math.max(y, 0), canvasHeight)

    predictedX.value = clampedX
    predictedY.value = clampedY

    // Visualize prediction
    if (ctx) {
      ctx.clearRect(0, 0, canvasWidth, canvasHeight)
      
      // Draw heat map first (background)
      drawHeatMap()
      
      // Draw flow field arrows
      drawFlowField()
      
      // Draw prediction point
      ctx.fillStyle = '#ff0000'
      ctx.beginPath()
      ctx.arc(clampedX, clampedY, 8, 0, Math.PI * 2)
      ctx.fill()
      
      // Draw confidence circle (based on loss)
      const confidence = Math.max(0.1, 1 - modelLoss.value)
      ctx.strokeStyle = `rgba(255, 0, 0, ${confidence})`
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(clampedX, clampedY, 20 * confidence, 0, Math.PI * 2)
      ctx.stroke()
      
      // Draw mouse position if tracking
      if (allowTracking) {
        ctx.fillStyle = '#00ff00'
        ctx.beginPath()
        ctx.arc(mouseX.value, mouseY.value, 4, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Clean up tensors
    input.dispose()
    output.dispose()

  } catch (err) {
    console.error('❌ Prediction error:', err)
  }
}

let trainIntervalId
let rafId

onMounted(async () => {
  try {
    tf = await initTF()
    await nextTick()

    // Get backend information
    currentBackend.value = getCurrentBackend()
    isWebGPU.value = isWebGPUAvailable()
    isWebGL.value = isWebGLAvailable()

    const cvs = canvas.value
    canvasWidth = 512
    canvasHeight = 512
    cvs.width = canvasWidth * devicePixelRatio
    cvs.height = canvasHeight * devicePixelRatio
    ctx = cvs.getContext('2d')
    ctx.scale(devicePixelRatio, devicePixelRatio)

    createModel()
    trainIntervalId = setInterval(trainModel, TRAIN_INTERVAL_MS)
    isActive = true

    const loop = async () => {
      if (isActive) {
        try {
          await predict()
        } catch (err) {
          console.error('Prediction error:', err)
        }
      }
      rafId = requestAnimationFrame(loop)
    }

    loop()
  } catch (err) {
    console.error('❌ Failed to initialize TensorFlow:', err)
    // Show user-friendly error message
    currentBackend.value = 'Failed'
    isActive = false
  }
})

onBeforeUnmount(() => {
  isActive = false
  clearInterval(trainIntervalId)
  cancelAnimationFrame(rafId)
  
  // Clean up TensorFlow tensors and model
  if (model) {
    model.dispose()
  }
  
  // Clean up tensor pool
  Object.values(tensorPool).forEach(tensor => {
    if (tensor) tensor.dispose()
  })
  
  // Clear training data
  trainingData = []
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
  cursor: crosshair;
}

.overlay {
  color: #fff;
  font-family: monospace;
  margin-top: 1rem;
  text-align: left;
  width: fit-content;
  margin: 0 auto;
  margin-bottom: 2rem;
  background: rgba(0, 0, 0, 0.8);
  padding: 1rem;
  border-radius: 8px;
}

.overlay h2 {
  color: #ff8800;
  margin-bottom: 1rem;
}

.overlay p {
  margin: 0.25rem 0;
  font-size: 0.9rem;
}

.controls {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 1rem;
  flex-wrap: wrap;
}

.control-btn {
  padding: 0.5rem 1rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  transition: background-color 0.2s;
}

.control-btn:hover {
  background: #0056b3;
}

.control-btn:active {
  transform: translateY(1px);
}

.note {
  background: #fff3cd;
  border: 1px solid #ffeaa7;
  border-radius: 8px;
  padding: 1rem;
  margin-top: 1rem;
}

.note p {
  margin: 0;
  color: #856404;
  font-size: 0.9rem;
}
</style>
