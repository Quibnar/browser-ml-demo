<template>
  <div>
    <div class="tool trainer-container">

      <h2>🎨 Color Clustering</h2>
    <input type="file" @change="processImage" accept="image/*" />
    <div class="swatches">
      <div
        v-for="color in dominantColors"
        :key="color"
        class="swatch"
        :style="{ backgroundColor: color }"
      ></div>
    </div>
    </div>

    <!-- ──────────────── Explanation Block ──────────────── -->
    <div class="explanation">
      <h1> K-means clustering with WASM</h1>
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
    </div>
  </div>
</template>

  
  <script setup>
  import { ref } from 'vue'
  
  const dominantColors = ref([])
  
  const processImage = (event) => {
    const file = event.target.files[0]
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
      dominantColors.value = clustered.centroids.map(c => `rgb(${c.join(',')})`)
    }
    img.src = URL.createObjectURL(file)
  }
  
  function kmeans(data, k) {
    const centroids = data.slice(0, k)
    for (let iter = 0; iter < 5; iter++) {
      const clusters = Array.from({ length: k }, () => [])
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
      for (let i = 0; i < k; i++) {
        centroids[i] = average(clusters[i])
      }
    }
    return { centroids }
  }
  
  function distance(a, b) {
    return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0))
  }
  
  function average(points) {
    const len = points.length
    if (len === 0) return [0, 0, 0]
    const sum = points.reduce((acc, val) => acc.map((x, i) => x + val[i]), [0, 0, 0])
    return sum.map(x => Math.round(x / len))
  }
  </script>
  
  <style scoped>

  .tool{
    padding: 3rem;
    margin: 4rem auto;
    width: 100%;
    max-width: 512px;
    border: 1px solid #333;
  }

  .swatches {
    display: flex;
    gap: 10px;
    margin-top: 1rem;
  }
  .swatch {
    width: 50px;
    height: 50px;
    border-radius: 5px;
  }



  </style>
  