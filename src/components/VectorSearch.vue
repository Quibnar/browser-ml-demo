<template>
    <div class="vector-search trainer-container">
        <h2>🔍 Vector Search (Semantic Similarity)</h2>

        <textarea v-model="query" placeholder="Enter your search query..." />
        <textarea v-model="candidatesText" placeholder="Enter candidate sentences (one per line)..." />

        <button @click="runSearch" :disabled="loading">
            {{ loading ? 'Searching...' : 'Search' }}
        </button>

        <ul v-if="results.length">
            <li v-for="(item, idx) in results" :key="idx">
                {{ item.text }} — <em>{{ item.similarity.toFixed(4) }}</em>
            </li>
        </ul>

        <div v-if="error" style="color: red;">
            <strong>Error:</strong> {{ error }}
        </div>
    </div>
<div class="explanation">
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

  <h3>\u{1f6e0️} Tech Details</h3>
  <p>Uses Transformers.js to run a quantized BERT-style model (MiniLM) directly in the browser. Embeddings are normalized and compared using cosine distance.</p>
</div>
<br>
</template>

<script setup>
import { ref } from 'vue'
import { pipeline, cos_sim, env } from '@xenova/transformers'

// Force CDN fetch
env.allowLocalModels  = false
env.useBrowserCache   = false


const query = ref('Which of these belong at a Circus?')
const candidatesText = ref(`Submarine
Mountain
Clown
Flower
Lion
Tightrope
Band`)
const results        = ref([])
const loading        = ref(false)
const error          = ref(null)

let embedder

async function runSearch() {
  loading.value   = true
  error.value     = null
  results.value   = []

  try {
    if (!embedder) {
      embedder = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
        { quantized: true }
      )
    }

    const candidates = candidatesText.value
      .split('\n')
      .map(s => s.trim())
      .filter(Boolean)

    if (!query.value || candidates.length === 0) {
      throw new Error('Please enter a query and at least one candidate sentence.')
    }

    // ─────────────────────────────
    // 1) Embed query + candidates
    // ─────────────────────────────
    // each call returns a Tensor with dims [N, 384]
    const qTensor = await embedder(query.value, {
      pooling: 'mean',
      normalize: true
    })
    const cTensor = await embedder(candidates, {
      pooling: 'mean',
      normalize: true
    })

    // 2) Convert to nested arrays
    const queryEmbedding      = qTensor.tolist()[0]     // [384 floats]
    const candidateEmbeddings = cTensor.tolist()        // [[384 floats], …]

    // 3) Compute cosine similarities
    const ranked = candidates.map((text, i) => ({
      text,
      similarity: cos_sim(queryEmbedding, candidateEmbeddings[i])
    }))

    // 4) Sort & display
    results.value = ranked.sort((a, b) => b.similarity - a.similarity)
  }
  catch (err) {
    console.error('❌ Error during vector search:', err)
    error.value = err.message || String(err)
  }
  finally {
    loading.value = false
  }
}

</script>

<style scoped>
.vector-search {
    padding: 2rem;
    font-family: sans-serif;
}

textarea {
    display: block;
    width: 100%;
    height: 100px;
    margin-bottom: 1rem;
    font-family: monospace;
    font-size: 1rem;
    padding: 0.5rem;
}

button {
    padding: 0.5rem 1rem;
    margin-bottom: 1rem;
}

ul {
    padding-left: 1rem;
}

li {
    margin-bottom: 0.5rem;
}
</style>