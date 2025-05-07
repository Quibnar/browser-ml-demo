<template>
  <div class="app-container">
    <select v-if="!isMobile" v-model="selectedDemo" class="select-menu">
      <option value="DemoIntro">Browser AI Demos</option>
      <option value="MouseTrainer">Behavior</option>
      <option value="ImageClassifier">Classification</option>
      <option value="ColorClustering">Summarization</option>
      <option value="SentimentAnalyzer">Analysis</option>
      <option value="VectorSearch">Relevance</option>
      <option value="MathExplainer">Utility</option>
    </select>

    <component :is="currentComponent" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'

import MouseTrainer from './components/MouseTrainer.vue'
import MathExplainer from './components/MathExplainer.vue'
import ImageClassifier from './components/ImageClassifier.vue'
import ColorClustering from './components/ColorClustering.vue'
import SentimentAnalyzer from './components/SentimentAnalyzer.vue'
import VectorSearch from './components/VectorSearch.vue'
import DemoIntro from './components/DemoIntro.vue'
import DeviceNotSupported from './components/DeviceNotSupported.vue'

const isMobile = ref(false)
const selectedDemo = ref('DemoIntro')

onMounted(() => {
  isMobile.value = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
})

const currentComponent = computed(() => {

  if (isMobile.value) return DeviceNotSupported

  switch (selectedDemo.value) {
    case 'MouseTrainer': return MouseTrainer
    case 'MathExplainer': return MathExplainer
    case 'ImageClassifier': return ImageClassifier
    case 'ColorClustering': return ColorClustering
    case 'SentimentAnalyzer': return SentimentAnalyzer
    case 'VectorSearch': return VectorSearch
    case 'DemoIntro': return DemoIntro
    default: return DemoIntro
  }
})
</script>

<style>
.app-container {
  padding: 0rem;
  font-family: sans-serif;
}

.select-menu {
  margin-bottom: 2rem;
  font-size: 1rem;
  position: fixed;
  top: 2em;
  left: 2em;
  padding: 0.5em;
  z-index: 100;
}
</style>
