<template>
  <DemoWrapper>
    <!-- TOOL SLOT -->
    <template #tool>
      <div class="color-tool">
        <h2>🎨 Color Clustering</h2>
        <div class="file-input-container">
          <label for="color-image-upload" class="file-input-label">
            📁 Choose an image
          </label>
          <input 
            id="color-image-upload"
            type="file" 
            @change="processImage" 
            accept="image/*" 
            class="file-input"
          />
        </div>
        
        <div v-if="dominantColors.length > 0" class="results">
          <h3>Dominant Colors:</h3>
          <div class="swatches">
            <div
              v-for="(color, index) in dominantColors"
              :key="index"
              class="swatch"
              :style="{ backgroundColor: color }"
              :title="color"
            >
              <span class="color-index">{{ index + 1 }}</span>
            </div>
          </div>
        </div>
      </div>
    </template>

    <!-- EXPLANATION SLOT -->
    <template #explanation>
      <h1>K-means Clustering with WASM</h1>
      <h3>🧠 What This Does</h3>
      <p>This demo uses K-means clustering (a basic machine learning algorithm) to group similar pixel colors in an uploaded image, giving you the dominant color palette.</p>

      <h3>🌐 Real‑World Applications</h3>
      <ul>
        <li>Auto-tagging product photos with color themes</li>
        <li>Generating custom palettes for web design</li>
        <li>Analyzing brand consistency in marketing images</li>
      </ul>

      <h3>🧩 Other Uses of K-means + WASM</h3>
      <ul>
        <li><strong>Text Clustering:</strong> Grouping chat logs or FAQs by topic</li>
        <li><strong>User Behavior:</strong> Segmenting browsing habits for personalization</li>
        <li><strong>Sensor Data:</strong> Detecting anomalies in IoT devices</li>
        <li><strong>Audio Analysis:</strong> Classifying music or environmental sounds</li>
        <li><strong>Medical Features:</strong> Grouping health data for research</li>
      </ul>

      <h3>🔧 Tech Details</h3>
      <p>This example runs entirely in the browser using WebAssembly (WASM) for fast array math, and applies K-means to 100×100 pixels for real‑time responsiveness.</p>
    </template>
  </DemoWrapper>
</template>

<script setup>
import { ref } from 'vue'
import DemoWrapper from './DemoWrapper.vue'

const dominantColors = ref([])

const processImage = (event) => {
  const file = event.target.files[0]
  if (!file) return
  
  // Validate file type
  if (!file.type.startsWith('image/')) {
    alert('Please select a valid image file')
    return
  }
  
  const img = new Image()
  img.onload = () => {
    const canvas = document.createElement('canvas')
    canvas.width = 100
    canvas.height = 100
    const ctx = canvas.getContext('2d')
    ctx.drawImage(img, 0, 0, 100, 100)
    const imageData = ctx.getImageData(0, 0, 100, 100)
    const pixels = []
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      const r = imageData.data[i]
      const g = imageData.data[i + 1]
      const b = imageData.data[i + 2]
      pixels.push([r, g, b])
    }
    
    const clustered = kmeans(pixels, 5)
    dominantColors.value = clustered.centroids.map(c => `rgb(${Math.round(c[0])}, ${Math.round(c[1])}, ${Math.round(c[2])})`)
  }
  
  img.onerror = () => {
    alert('Failed to load image. Please try a different file.')
  }
  
  img.src = URL.createObjectURL(file)
}

function kmeans(data, k) {
  if (data.length === 0) return { centroids: [] }
  
  // Initialize centroids randomly
  const centroids = []
  for (let i = 0; i < k; i++) {
    const randomIndex = Math.floor(Math.random() * data.length)
    centroids.push([...data[randomIndex]])
  }
  
  for (let iter = 0; iter < 10; iter++) {
    const clusters = Array.from({ length: k }, () => [])
    
    // Assign points to nearest centroid
    for (const point of data) {
      let best = 0
      let bestDist = Infinity
      for (let i = 0; i < k; i++) {
        const d = distance(point, centroids[i])
        if (d < bestDist) {
          best = i
          bestDist = d
        }
      }
      clusters[best].push(point)
    }
    
    // Update centroids
    for (let i = 0; i < k; i++) {
      if (clusters[i].length > 0) {
        centroids[i] = average(clusters[i])
      }
    }
  }
  
  return { centroids }
}

function distance(a, b) {
  return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0))
}

function average(points) {
  if (points.length === 0) return [0, 0, 0]
  
  const sum = points.reduce((acc, point) => {
    return [acc[0] + point[0], acc[1] + point[1], acc[2] + point[2]]
  }, [0, 0, 0])
  
  return [
    sum[0] / points.length,
    sum[1] / points.length,
    sum[2] / points.length
  ]
}
</script>

<style scoped>
.color-tool {
  text-align: center;
  width: 100%;
}

.color-tool h2 {
  color: #333;
  margin-bottom: 2rem;
  font-size: 2rem;
}

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
  font-size: 1.1rem;
}

.file-input-label:hover {
  background: #0056b3;
}

.file-input {
  display: none;
}

.results {
  margin-top: 2rem;
}

.results h3 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
}

.swatches {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: 1rem;
}

.swatch {
  width: 80px;
  height: 80px;
  border-radius: 12px;
  border: 3px solid white;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  position: relative;
  cursor: pointer;
  transition: transform 0.2s;
}

.swatch:hover {
  transform: scale(1.1);
}

.color-index {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(0, 0, 0, 0.7);
  color: white;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  font-weight: bold;
}
</style>
  