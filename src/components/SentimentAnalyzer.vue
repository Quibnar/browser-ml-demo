<template>
  <DemoWrapper>
    <!-- TOOL SLOT -->
    <template #tool>
      <div class="sentiment-tool">
        <h2>📝 Sentiment Analyzer</h2>
        <div class="input-section">
          <textarea 
            v-model="inputText" 
            placeholder="Type a sentence to analyze its sentiment..." 
            class="text-input"
          />
          <button 
            @click="analyzeSentiment" 
            :disabled="loading"
            class="analyze-button"
          >
            {{ loading ? '🔍 Analyzing...' : '🧠 Analyze Sentiment' }}
          </button>
        </div>
        
        <div v-if="result.label && !error" class="result-section">
          <h3>Analysis Result:</h3>
          <div class="sentiment-display">
            <div class="sentiment-label">
              <span class="label-text">{{ result.label }}</span>
              <span class="confidence-score">{{ (result.score * 100).toFixed(1) }}% confidence</span>
            </div>
          </div>
        </div>
        
        <div v-if="error" class="error-section">
          <h3>Error:</h3>
          <p>{{ error }}</p>
          <button @click="retryAnalysis" class="retry-button">🔄 Retry</button>
        </div>
      </div>
    </template>

    <!-- EXPLANATION SLOT -->
    <template #explanation>
      <h1>Sentiment Analysis</h1>
      <h2>with TensorFlow.js + Browser Embeddings</h2>
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

      <h3>🛠️ Tech Details</h3>
      <p>Powered by TensorFlow.js with a small sentiment dataset. All training and inference happen locally, making it private and responsive.</p>
    </template>
  </DemoWrapper>
</template>

<script setup>
import { ref } from 'vue'
import DemoWrapper from './DemoWrapper.vue'
// import both pipeline _and_ env
import { pipeline, env } from '@xenova/transformers'

// disable local file loading (and clear stale cache if needed)
env.allowLocalModels = false
env.useBrowserCache = false

const inputText = ref('This Rocks!')
const result = ref({ label: '', score: 0 })
const loading = ref(false)
const error = ref(null)

let classifier

const retryAnalysis = () => {
  error.value = null
  result.value = { label: '', score: 0 }
}

async function analyzeSentiment() {
  if (!inputText.value.trim()) {
    error.value = 'Please enter some text to analyze'
    return
  }
  
  loading.value = true
  error.value = null
  
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
    error.value = String(err)
    result.value = { label: 'Error', score: 0 }
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.sentiment-tool {
  text-align: center;
  width: 100%;
}

.sentiment-tool h2 {
  color: #333;
  margin-bottom: 2rem;
  font-size: 2rem;
}

.input-section {
  margin-bottom: 2rem;
}

.text-input {
  width: 100%;
  height: 120px;
  font-family: inherit;
  font-size: 1rem;
  padding: 1rem;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  resize: vertical;
  margin-bottom: 1rem;
  transition: border-color 0.2s;
}

.text-input:focus {
  outline: none;
  border-color: #007bff;
}

.analyze-button {
  padding: 1rem 2rem;
  font-size: 1.1rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
  font-family: inherit;
}

.analyze-button:hover:not(:disabled) {
  background: #0056b3;
}

.analyze-button:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.result-section {
  margin-top: 2rem;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 12px;
  border: 1px solid #e9ecef;
}

.result-section h3 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
}

.sentiment-display {
  display: flex;
  justify-content: center;
}

.sentiment-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.label-text {
  font-size: 1.5rem;
  font-weight: bold;
  color: #28a745;
  text-transform: capitalize;
}

.confidence-score {
  font-size: 1rem;
  color: #6c757d;
}

.error-section {
  margin-top: 2rem;
  padding: 1.5rem;
  background: #f8d7da;
  border: 1px solid #f5c6cb;
  border-radius: 12px;
  color: #721c24;
}

.error-section h3 {
  margin-bottom: 1rem;
  font-size: 1.3rem;
}

.retry-button {
  margin-top: 1rem;
  padding: 0.5rem 1rem;
  background: #dc3545;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
}

.retry-button:hover {
  background: #c82333;
}
</style>