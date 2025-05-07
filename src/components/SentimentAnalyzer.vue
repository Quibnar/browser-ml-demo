<template>
    <div class="sentiment-container  trainer-container">
        <h2>📝 Sentiment Analyzer</h2>
        <textarea v-model="inputText" placeholder="Type a sentence..." />
        <button @click="analyzeSentiment" :disabled="loading">
            {{ loading ? 'Analyzing...' : 'Analyze Sentiment' }}
        </button>
        <div v-if="result.label">
            <p><strong>Sentiment:</strong> {{ result.label }}</p>
            <p><strong>Confidence:</strong> {{ (result.score * 100).toFixed(2) }}%</p>
        </div>
        <div v-if="error">
            <p style="color: red;"><strong>Error:</strong> {{ error }}</p>
        </div>
    </div>
    <h1>Sentiment Analysis</h1>
<h2>with TensorFlow.js + Browser Embeddings</h2>
<div class="explanation">
  <h3>📖 What This Does</h3>
  <p>This tool performs simple sentiment detection (positive/negative) on input text using a locally trained model. It uses tokenization and embedding layers processed in-browser.</p>

  <h3>🌐 Real‑World Applications</h3>
  <ul>
    <li>Customer feedback parsing</li>
    <li>Content moderation and toxicity flags</li>
    <li>Social media trend analysis</li>
  </ul>

  <h3>🔧 Broader Uses for Text Classification</h3>
  <ul>
    <li><strong>Chatbots:</strong> Detect tone and mood</li>
    <li><strong>Education:</strong> Score essay or review quality</li>
    <li><strong>News:</strong> Gauge polarity of coverage</li>
  </ul>

  <h3>\u{1f6e0️} Tech Details</h3>
  <p>Powered by TensorFlow.js with a small sentiment dataset. All training and inference happen locally, making it private and responsive.</p>
</div>
</template>

<script setup>
import { ref } from 'vue'
// import both pipeline _and_ env
import { pipeline, env } from '@xenova/transformers'

// disable local file loading (and clear stale cache if needed)
env.allowLocalModels = false
env.useBrowserCache   = false

const inputText = ref('This Rocks!')
const result    = ref({ label: '', score: 0 })
const loading   = ref(false)
const error     = ref(null)

let classifier

async function analyzeSentiment() {
  loading.value = true
  error.value   = null
  try {
    if (!classifier) {
      classifier = await pipeline(
        'sentiment-analysis',
        'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
        {
          dtype: 'q8',               // use quantized 8‑bit for WASM
          progress_callback: console.log
        }
      )
    }
    const [output] = await classifier(inputText.value)
    result.value = output
  } catch (err) {
    console.error('❌ Error running sentiment analysis:', err)
    error.value  = String(err)
    result.value = { label: 'Error', score: 0 }
  } finally {
    loading.value = false
  }
}
</script>


<style scoped>
.sentiment-container {
    padding: 2rem;
    font-family: sans-serif;
}

textarea {
    width: 100%;
    height: 100px;
    font-family: monospace;
    font-size: 1rem;
    padding: 0.5rem;
}

button {
    margin-top: 1rem;
    padding: 0.5rem 1rem;
}
</style>