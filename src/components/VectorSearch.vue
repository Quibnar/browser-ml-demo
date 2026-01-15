<template>
  <DemoWrapper>
    <!-- TOOL SLOT -->
    <template #tool>
      <div class="vector-search-tool">
        <h2>🔍 Vector Search (Semantic Similarity)</h2>

        <div class="input-section">
          <div class="input-group">
            <label for="query-input">Search Query:</label>
            <textarea 
              id="query-input"
              v-model="query" 
              placeholder="Enter your search query..." 
              class="text-input"
            />
          </div>
          
          <div class="input-group">
            <label for="candidates-input">Candidate Sentences:</label>
            <textarea 
              id="candidates-input"
              v-model="candidatesText" 
              placeholder="Enter candidate sentences (one per line)..." 
              class="text-input"
            />
          </div>
        </div>

        <button @click="runSearch" :disabled="loading" class="search-button">
          {{ loading ? '🔍 Searching...' : '🔍 Run Search' }}
        </button>

        <div v-if="results.length > 0" class="results-section">
          <h3>Search Results (ranked by similarity):</h3>
          <ul class="results-list">
            <li v-for="(item, idx) in results" :key="idx" class="result-item">
              <span class="result-text">{{ item.text }}</span>
              <span class="similarity-score">{{ item.similarity.toFixed(4) }}</span>
            </li>
          </ul>
        </div>

        <div v-if="error" class="error-section">
          <h3>Error:</h3>
          <p>{{ error }}</p>
          <button @click="clearError" class="retry-button">🔄 Clear Error</button>
        </div>
      </div>
    </template>

    <!-- EXPLANATION SLOT -->
    <template #explanation>
      <h1>Semantic Relevance</h1>
      <h2>with Transformers.js</h2>
      <h3>📖 What This Does</h3>
      <p>This demo compares a user query with multiple candidate sentences using semantic embeddings. It ranks the candidates by cosine similarity to the query.</p>

      <h3>🌐 Real‑World Applications</h3>
      <ul>
        <li>FAQ and knowledge base search</li>
        <li>Smart autocomplete or suggestion engines</li>
        <li>Conversational memory or context-matching</li>
      </ul>

      <h3>🔧 Broader Uses for Embedding Similarity</h3>
      <ul>
        <li><strong>Education:</strong> Match questions to answers</li>
        <li><strong>Legal:</strong> Search contracts by meaning</li>
        <li><strong>Support:</strong> Surface past relevant cases</li>
      </ul>

      <h3>🛠️ Tech Details</h3>
      <p>Uses Transformers.js to run a quantized BERT-style model (MiniLM) directly in the browser. Embeddings are normalized and compared using cosine distance.</p>
    </template>
  </DemoWrapper>
</template>

<script setup>
import { ref } from 'vue'
import DemoWrapper from './DemoWrapper.vue'
import { pipeline, env } from '@xenova/transformers'

// Force CDN fetch
env.allowLocalModels = false
env.useBrowserCache = false

const query = ref('Which of these belong at a Circus?')
const candidatesText = ref(`Submarine
Mountain
Clown
Flower
Lion
Tightrope
Band`)
const results = ref([])
const loading = ref(false)
const error = ref(null)

let embedder

const clearError = () => {
  error.value = null
}

// Helper function to compute cosine similarity between two arrays
function cosineSimilarity(a, b) {
  if (a.length !== b.length) return 0
  
  let dotProduct = 0
  let normA = 0
  let normB = 0
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  
  if (normA === 0 || normB === 0) return 0
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
}

async function runSearch() {
  if (!query.value.trim()) {
    error.value = 'Please enter a search query'
    return
  }
  
  const candidates = candidatesText.value
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean)
    
  if (candidates.length === 0) {
    error.value = 'Please enter at least one candidate sentence'
    return
  }

  loading.value = true
  error.value = null
  results.value = []

  try {
    if (!embedder) {
      embedder = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
        { quantized: true }
      )
    }

    // ─────────────────────────────
    // 1) Embed query + candidates
    // ─────────────────────────────
    const qTensor = await embedder(query.value, {
      pooling: 'mean',
      normalize: true
    })

    const candidateTensors = await Promise.all(
      candidates.map(c => embedder(c, { pooling: 'mean', normalize: true }))
    )

    // ─────────────────────────────
    // 2) Convert tensors to arrays and compute similarities
    // ─────────────────────────────
    const queryEmbedding = Array.from(qTensor.data)
    const similarities = candidateTensors.map(cTensor => {
      const candidateEmbedding = Array.from(cTensor.data)
      return cosineSimilarity(queryEmbedding, candidateEmbedding)
    })

    // ─────────────────────────────
    // 3) Rank results
    // ─────────────────────────────
    const ranked = candidates.map((text, idx) => ({
      text,
      similarity: similarities[idx]
    })).sort((a, b) => b.similarity - a.similarity)

    results.value = ranked

    // Note: Transformers.js handles memory management automatically
    // No need to manually dispose tensors

  } catch (err) {
    console.error('❌ Vector search error:', err)
    error.value = String(err)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.vector-search-tool {
  text-align: center;
  width: 100%;
}

.vector-search-tool h2 {
  color: #333;
  margin-bottom: 2rem;
  font-size: 2rem;
}

.input-section {
  margin-bottom: 2rem;
  text-align: left;
}

.input-group {
  margin-bottom: 1.5rem;
}

.input-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #333;
  text-align: left;
}

.text-input {
  width: 100%;
  height: 80px;
  font-family: inherit;
  font-size: 1rem;
  padding: 0.75rem;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  resize: vertical;
  transition: border-color 0.2s;
}

.text-input:focus {
  outline: none;
  border-color: #007bff;
}

.search-button {
  padding: 1rem 2rem;
  font-size: 1.1rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
  font-family: inherit;
  margin-bottom: 2rem;
}

.search-button:hover:not(:disabled) {
  background: #0056b3;
}

.search-button:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.results-section {
  margin-top: 2rem;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 12px;
  border: 1px solid #e9ecef;
  text-align: left;
}

.results-section h3 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
  text-align: center;
}

.results-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.result-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.result-text {
  flex: 1;
  text-align: left;
  color: #333;
}

.similarity-score {
  font-weight: bold;
  color: #007bff;
  font-family: monospace;
  background: #e3f2fd;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
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