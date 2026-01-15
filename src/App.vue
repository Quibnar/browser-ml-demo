<template>
  <div class="app-container">
    <nav v-if="!isMobile" class="demo-navigation" role="navigation" aria-label="Demo selection">
      <h2 class="demo-title">Browser AI Demos</h2>
      <ul class="demo-menu" role="list">
        <li v-for="demo in demos" :key="demo.value" class="demo-item">
          <button 
            :class="['demo-button', { active: selectedDemo === demo.value }]"
            @click="selectedDemo = demo.value"
            :aria-current="selectedDemo === demo.value ? 'page' : undefined"
            :aria-label="`Switch to ${demo.label} demo`"
          >
            <span class="demo-icon">{{ demo.icon }}</span>
            <span class="demo-label">{{ demo.label }}</span>
          </button>
        </li>
      </ul>
    </nav>

    <main class="main-content" role="main">
      <component :is="currentComponent" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { defineAsyncComponent } from 'vue'

// Lazy‐load each demo only when it's actually rendered
const DemoIntro        = defineAsyncComponent(() => import('./components/DemoIntro.vue'))
const MouseTrainer     = defineAsyncComponent(() => import('./components/MouseTrainer.vue'))
const ImageClassifier  = defineAsyncComponent(() => import('./components/ImageClassifier.vue'))
const ColorClustering  = defineAsyncComponent(() => import('./components/ColorClustering.vue'))
const SentimentAnalyzer= defineAsyncComponent(() => import('./components/SentimentAnalyzer.vue'))
const VectorSearch     = defineAsyncComponent(() => import('./components/VectorSearch.vue'))
const MathExplainer    = defineAsyncComponent(() => import('./components/MathExplainer.vue'))
const BatchVisualizer  = defineAsyncComponent(() => import('./components/BatchVisualizer.vue'))
const DigitTrainer     = defineAsyncComponent(() => import('./components/DigitTrainer.vue'))
const DeviceNotSupported = defineAsyncComponent(() => import('./components/DeviceNotSupported.vue'))

const isMobile = ref(false)
const selectedDemo = ref('DemoIntro')

// Demo configuration with icons and labels
const demos = [
  { value: 'DemoIntro', label: 'Introduction', icon: '🏠' },
  { value: 'MouseTrainer', label: 'Behavior Modeling', icon: '🧠' },
  { value: 'ImageClassifier', label: 'Image Classification', icon: '🖼️' },
  { value: 'ColorClustering', label: 'Color Clustering', icon: '🎨' },
  { value: 'SentimentAnalyzer', label: 'Sentiment Analysis', icon: '😊' },
  { value: 'VectorSearch', label: 'Vector Search', icon: '🔍' },
  { value: 'MathExplainer', label: 'Math Explainer', icon: '📐' },
  { value: 'BatchVisualizer', label: 'Batch Training', icon: '⚡' },
  { value: 'DigitTrainer', label: 'Live Digit Trainer', icon: '✏️' }
]

onMounted(() => {
  // Better mobile detection
  isMobile.value = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i
    .test(navigator.userAgent) || window.innerWidth <= 768
})

const currentComponent = computed(() => {
  switch (selectedDemo.value) {
    case 'MouseTrainer':      return MouseTrainer
    case 'ImageClassifier':   return ImageClassifier
    case 'ColorClustering':   return ColorClustering
    case 'SentimentAnalyzer': return SentimentAnalyzer
    case 'VectorSearch':      return VectorSearch
    case 'MathExplainer':     return MathExplainer
    case 'BatchVisualizer':   return BatchVisualizer
    case 'DigitTrainer':      return DigitTrainer
    case 'DemoIntro':         return DemoIntro
    default:                  return DemoIntro
  }
})
</script>

<style scoped>
.app-container {
  flex: 1;
  display: flex;
  flex-direction: row;
  padding: 0;
  text-align: left;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  height: 100vh;
  background: #f8f9fa;
}

.demo-navigation {
  width: 280px;
  background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
  padding: 2rem 1.5rem;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
  position: fixed;
  top: 0;
  left: 0;
  height: 100%;
  z-index: 100;
  overflow-y: auto;
}

.demo-title {
  font-size: 1.8rem;
  margin-bottom: 2rem;
  color: #ecf0f1;
  font-weight: 600;
  text-align: center;
  letter-spacing: 0.5px;
}

.demo-menu {
  width: 100%;
  list-style: none;
  padding: 0;
  margin: 0;
}

.demo-item {
  margin-bottom: 0.75rem;
  width: 100%;
}

.demo-button {
  width: 100%;
  padding: 1rem 1.25rem;
  text-align: left;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: all 0.2s ease;
  color: #ecf0f1;
  font-size: 0.95rem;
  font-weight: 500;
}

.demo-button:hover {
  background: rgba(255, 255, 255, 0.15);
  border-color: rgba(255, 255, 255, 0.3);
  transform: translateX(4px);
}

.demo-button.active {
  background: #3498db;
  border-color: #3498db;
  color: white;
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
}

.demo-button:focus {
  outline: none;
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.3);
}

.demo-icon {
  font-size: 1.3rem;
  margin-right: 12px;
  width: 24px;
  text-align: center;
}

.demo-label {
  font-size: 0.95rem;
  font-weight: 500;
}

.main-content {
  flex-grow: 1;
  padding: 2rem;
  margin-left: 350px;
  background: #f8f9fa;
  overflow-y: auto;
}

/* Custom scrollbar for navigation */
.demo-navigation::-webkit-scrollbar {
  width: 6px;
}

.demo-navigation::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.demo-navigation::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 3px;
}

.demo-navigation::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.5);
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .app-container {
    flex-direction: column;
    height: auto;
  }

  .demo-navigation {
    width: 100%;
    height: auto;
    position: relative;
    flex-direction: row;
    justify-content: space-around;
    padding: 1rem 0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  }

  .demo-title {
    display: none;
  }

  .demo-menu {
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .demo-item {
    margin-bottom: 0;
    width: auto;
  }

  .demo-button {
    width: auto;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    white-space: nowrap;
  }

  .demo-button:hover {
    transform: translateY(-2px);
  }

  .demo-icon {
    font-size: 1.1rem;
    margin-right: 8px;
    width: 20px;
  }

  .main-content {
    margin-left: 0;
    padding: 1rem;
  }
}

@media (max-width: 480px) {
  .demo-navigation {
    padding: 0.75rem 0;
  }

  .demo-button {
    padding: 0.5rem 0.75rem;
    font-size: 0.8rem;
  }

  .demo-icon {
    font-size: 1rem;
    margin-right: 6px;
    width: 18px;
  }

  .main-content {
    padding: 0.75rem;
  }
}
</style>
