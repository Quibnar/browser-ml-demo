<template>
  <DemoWrapper>
    <!-- TOOL SLOT -->
    <template #tool>
      <div class="math-tool">
        <h2>🧠 Math Explainer</h2>
        <div class="input-section">
          <input 
            v-model="input" 
            placeholder="Enter an expression or equation..." 
            class="math-input"
          />
          <button @click="analyze" class="analyze-button">🔍 Analyze</button>
        </div>

        <div v-if="steps.length > 0" class="results-section">
          <h3>📐 Math.js Steps:</h3>
          <ul class="steps-list">
            <li v-for="(step, idx) in steps" :key="idx" class="step-item">{{ step }}</li>
          </ul>
        </div>

        <div v-if="nerdamerSteps.length > 0" class="results-section">
          <h3>🔬 Nerdamer Results:</h3>
          <ul class="steps-list">
            <li v-for="(item, idx) in nerdamerSteps" :key="'n' + idx" class="step-item">{{ item }}</li>
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
      <h1>Utility</h1>
      <h2>with Math.js + Nerdamer</h2>
      <h3>📖 What This Does</h3>
      <p>This module parses and simplifies math expressions using two symbolic algebra engines. Math.js shows
        step-by-step transformation, while Nerdamer provides symbolic expansions and solutions.</p>

      <h3>🌐 Real‑World Applications</h3>
      <ul>
        <li>Math tutoring and homework assistance</li>
        <li>Expression validation or simplification in apps</li>
        <li>Building computational notebooks or symbolic tools</li>
      </ul>

      <h3>🔧 Broader Uses for Symbolic Reasoning</h3>
      <ul>
        <li><strong>STEM Education:</strong> Explain transformations</li>
        <li><strong>Scientific Tools:</strong> Parse formulas interactively</li>
        <li><strong>Math Input:</strong> Clean or reformat math in real time</li>
      </ul>

      <h3>🛠️ Tech Details</h3>
      <p>Math.js runs step breakdowns in JavaScript. Nerdamer solves and expands symbolic equations. All logic is
        local and runs on small algebraic inputs.</p>

      <h2>Why Not GPT? Browser-Based ML Alternatives</h2>
      <p>
        General-purpose LLMs like GPT are powerful, but often unnecessary for focused tasks like classification,
        prediction, or semantic similarity.
        The following tools provide lightweight, fast, and private ML solutions that run entirely in your browser.
      </p>

      <ul>
        <li><strong>General ML Frameworks</strong>
          <ul>
            <li><a href="https://www.tensorflow.org/js" target="_blank" rel="noopener">TensorFlow.js</a> — Full-featured deep
              learning framework for browser-based training and inference.</li>
            <li><a href="https://ml5js.org/" target="_blank" rel="noopener">ml5.js</a> — Beginner-friendly ML library built on
              top of TensorFlow.js, with pretrained models for images, text, and more.</li>
          </ul>
        </li>

        <li><strong>Transformer Models</strong>
          <ul>
            <li><a href="https://github.com/xenova/transformers.js" target="_blank" rel="noopener">Transformers.js</a> —
              Hugging Face-compatible transformer models (BERT, CLIP, Whisper, etc.) for browser inference via
              WebAssembly.</li>
            <li><a href="https://github.com/xenova/onnxruntime-web" target="_blank" rel="noopener">ONNX Runtime Web</a> —
              Executes ONNX models (image, audio, language) with WASM/WebGL backends.</li>
          </ul>
        </li>

        <li><strong>Audio / Voice</strong>
          <ul>
            <li><a href="https://github.com/Picovoice/porcupine" target="_blank" rel="noopener">Porcupine</a> — Wake word
              detection entirely in the browser, using a compact neural model.</li>
            <li><a href="https://github.com/Picovoice/web-voice-processor" target="_blank" rel="noopener">Web Voice
                Processor</a> — Real-time audio input for ML-powered voice tools.</li>
          </ul>
        </li>

        <li><strong>Vision / Classification</strong>
          <ul>
            <li><a href="https://github.com/ml5js/ml5-library/tree/main/src/ImageClassifier" target="_blank" rel="noopener">ml5
                ImageClassifier</a> — Run MobileNet and other classifiers directly in the browser with no
              server dependency.</li>
            <li><a href="https://github.com/xenova/transformers.js/tree/main/examples/webcam"
                target="_blank" rel="noopener">Transformers.js CLIP</a> — Match text to images using a vision-language
              transformer model.</li>
          </ul>
        </li>
      </ul>

      <p>
        These tools demonstrate that you can build powerful AI experiences without LLMs, servers, or subscriptions —
        just pure browser-based machine learning.
      </p>
    </template>
  </DemoWrapper>
</template>

<script setup>
import { ref } from 'vue'
import DemoWrapper from './DemoWrapper.vue'
import * as math from 'mathjs'
import nerdamer from 'nerdamer'
import 'nerdamer/all'

const input = ref('2x + 3x')
const steps = ref([])
const nerdamerSteps = ref([])
const error = ref(null)

const clearError = () => {
  error.value = null
}

const analyze = () => {
  if (!input.value.trim()) {
    error.value = 'Please enter a mathematical expression'
    return
  }

  try {
    error.value = null
    steps.value = []
    nerdamerSteps.value = []

    // Math.js analysis
    try {
      // Check if the expression contains variables
      if (input.value.includes('x') || input.value.includes('y') || input.value.includes('z')) {
        // For expressions with variables, try to simplify
        const simplified = math.simplify(input.value)
        steps.value.push(`Original: ${input.value}`)
        steps.value.push(`Simplified: ${simplified.toString()}`)
      } else {
        // For numeric expressions, evaluate
        const result = math.evaluate(input.value)
        steps.value.push(`Result: ${result}`)
      }
    } catch (mathErr) {
      steps.value.push(`Math.js error: ${mathErr.message}`)
    }

    // Nerdamer analysis
    try {
      const parsed = nerdamer(input.value)
      nerdamerSteps.value.push(`Parsed: ${parsed.toString()}`)
      
      // Try to expand
      try {
        const expanded = parsed.expand()
        if (expanded.toString() !== parsed.toString()) {
          nerdamerSteps.value.push(`Expanded: ${expanded.toString()}`)
        }
      } catch (expandErr) {
        nerdamerSteps.value.push(`Expand: ${parsed.toString()} (already in simplest form)`)
      }
      
      // Try to factor
      try {
        const factored = parsed.factor()
        if (factored.toString() !== parsed.toString()) {
          nerdamerSteps.value.push(`Factored: ${factored.toString()}`)
        }
      } catch (factorErr) {
        nerdamerSteps.value.push(`Factor: ${parsed.toString()} (cannot be factored further)`)
      }

      // Try to solve if it's an equation
      if (input.value.includes('=')) {
        try {
          const solutions = nerdamer.solve(input.value, 'x')
          if (solutions.length > 0) {
            nerdamerSteps.value.push(`Solutions for x: ${solutions.toString()}`)
          } else {
            nerdamerSteps.value.push(`No solutions found for x`)
          }
        } catch (solveErr) {
          nerdamerSteps.value.push(`Solve: Cannot solve this equation`)
        }
      }
    } catch (nerdamerErr) {
      nerdamerSteps.value.push(`Nerdamer error: ${nerdamerErr.message}`)
    }

  } catch (err) {
    error.value = `Analysis failed: ${err.message}`
  }
}
</script>

<style scoped>
.math-tool {
  text-align: center;
  width: 100%;
}

.math-tool h2 {
  color: #333;
  margin-bottom: 2rem;
  font-size: 2rem;
}

.input-section {
  margin-bottom: 2rem;
  display: flex;
  gap: 1rem;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
}

.math-input {
  padding: 0.75rem 1rem;
  font-size: 1rem;
  border: 2px solid #e1e5e9;
  border-radius: 8px;
  min-width: 300px;
  font-family: 'Courier New', monospace;
  transition: border-color 0.2s;
}

.math-input:focus {
  outline: none;
  border-color: #007bff;
}

.analyze-button {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
  font-family: inherit;
}

.analyze-button:hover {
  background: #0056b3;
}

.results-section {
  margin: 2rem 0;
  padding: 1.5rem;
  background: #f8f9fa;
  border-radius: 12px;
  border: 1px solid #e9ecef;
  color: #333;
  text-align: left;
}

.results-section h3 {
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.3rem;
  text-align: center;
}

.steps-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.step-item {
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  font-family: 'Courier New', monospace;
  font-size: 0.9rem;
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

@media (max-width: 768px) {
  .input-section {
    flex-direction: column;
    align-items: stretch;
  }
  
  .math-input {
    min-width: auto;
    width: 100%;
  }
}
</style>