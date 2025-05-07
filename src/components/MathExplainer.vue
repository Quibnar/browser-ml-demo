<template>
    <div class="math-explainer trainer-container">
        <div class="">
            <h2>🧠 Math Explainer</h2>
            <input v-model="input" placeholder="Enter an expression or equation..." />
            <button @click="analyze">Analyze</button>
        </div>

        <div v-if="steps.length">
            <h3>📐 Math.js Steps:</h3>
            <ul>
                <li v-for="(step, idx) in steps" :key="idx">{{ step }}</li>
            </ul>
        </div>

        <div v-if="nerdamerSteps.length">
            <h3>🔬 Nerdamer Results:</h3>
            <ul>
                <li v-for="(item, idx) in nerdamerSteps" :key="'n' + idx">{{ item }}</li>
            </ul>
        </div>

        <p v-if="error" style="color:red;"><strong>Error:</strong> {{ error }}</p>
    </div>

    <h1>Utility</h1>
    <h2>with Math.js + Nerdamer</h2>
    <div class="explanation">
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

        <h3>\u{1f6e0️} Tech Details</h3>
        <p>Math.js runs step breakdowns in JavaScript. Nerdamer solves and expands symbolic equations. All logic is
            local and runs on small algebraic inputs.</p>
<br><br><br>
        <h2>Why Not GPT? Browser-Based ML Alternatives</h2>
        <p>
            General-purpose LLMs like GPT are powerful, but often unnecessary for focused tasks like classification,
            prediction, or semantic similarity.
            The following tools provide lightweight, fast, and private ML solutions that run entirely in your browser.
        </p>

        <ul>
            <li><strong>General ML Frameworks</strong>
                <ul>
                    <li><a href="https://www.tensorflow.org/js" target="_blank">TensorFlow.js</a> — Full-featured deep
                        learning framework for browser-based training and inference.</li>
                    <li><a href="https://ml5js.org/" target="_blank">ml5.js</a> — Beginner-friendly ML library built on
                        top of TensorFlow.js, with pretrained models for images, text, and more.</li>
                </ul>
            </li>

            <li><strong>Transformer Models</strong>
                <ul>
                    <li><a href="https://github.com/xenova/transformers.js" target="_blank">Transformers.js</a> —
                        Hugging Face-compatible transformer models (BERT, CLIP, Whisper, etc.) for browser inference via
                        WebAssembly.</li>
                    <li><a href="https://github.com/xenova/onnxruntime-web" target="_blank">ONNX Runtime Web</a> —
                        Executes ONNX models (image, audio, language) with WASM/WebGL backends.</li>
                </ul>
            </li>

            <li><strong>Audio / Voice</strong>
                <ul>
                    <li><a href="https://github.com/Picovoice/porcupine" target="_blank">Porcupine</a> — Wake word
                        detection entirely in the browser, using a compact neural model.</li>
                    <li><a href="https://github.com/Picovoice/web-voice-processor" target="_blank">Web Voice
                            Processor</a> — Real-time audio input for ML-powered voice tools.</li>
                </ul>
            </li>

            <li><strong>Vision / Classification</strong>
                <ul>
                    <li><a href="https://github.com/ml5js/ml5-library/tree/main/src/ImageClassifier" target="_blank">ml5
                            ImageClassifier</a> — Run MobileNet and other classifiers directly in the browser with no
                        server dependency.</li>
                    <li><a href="https://github.com/xenova/transformers.js/tree/main/examples/webcam"
                            target="_blank">Transformers.js CLIP</a> — Match text to images using a vision-language
                        transformer model.</li>
                </ul>
            </li>
        </ul>

        <p>
            These tools demonstrate that you can build powerful AI experiences without LLMs, servers, or subscriptions —
            just browser-native intelligence.
        </p>
        <br><br><br><br><br><br><br><br><br>
    </div>

</template>

<script setup>
import { ref } from 'vue'
import { simplify, parse, format } from 'mathjs'
import nerdamer from 'nerdamer/all.min'
import 'katex/dist/katex.min.css'

const input = ref('x = 2x + y^(x/y)')
const steps = ref([])
const nerdamerSteps = ref([])
const error = ref(null)

function analyze() {
    steps.value = []
    nerdamerSteps.value = []
    error.value = null

    try {
        let rhs = input.value
        let isEquation = false

        // Handle math.js simplification on RHS only
        if (input.value.includes('=')) {
            isEquation = true
            rhs = input.value.split('=').slice(1).join('=').trim()
            steps.value.push(`Parsed right-hand side: ${rhs}`)
        }

        // Parse and simplify with math.js
        const node = parse(rhs)
        steps.value.push(`Original: ${format(node)}`)
        const simplified = simplify(node)
        steps.value.push(`Simplified: ${format(simplified)}`)

        // Walk the tree
        node.traverse((child, path, parent) => {
            try {
                const simp = simplify(child)
                if (!child.equals(simp)) {
                    steps.value.push(`${format(child)} ⟶ ${format(simp)}`)
                } else {
                    steps.value.push(`${format(child)} ⟶ ${format(child)}`)
                }
            } catch { }
        })

        // Nerdamer parsing
        const nerd = nerdamer(input.value)
        nerdamerSteps.value.push(`Original: ${nerd.toString()}`)

        try {
            const expanded = nerd.expand().toString()
            if (expanded !== nerd.toString()) {
                nerdamerSteps.value.push(`Expanded: ${expanded}`)
            } else {
                nerdamerSteps.value.push(`Expanded: ${expanded}`) // Still show
            }
        } catch (e) {
            nerdamerSteps.value.push(`⚠️ Nerdamer expand error: ${e.message}`)
        }

        // Nerdamer solve (only if input is equation)
        if (isEquation) {
            try {
                const solutions = nerdamer.solve(input.value, 'x')
                nerdamerSteps.value.push(`Solved for x: ${solutions.toString()}`)
            } catch (e) {
                nerdamerSteps.value.push(`⚠️ Nerdamer solve error: ${e.message}`)
            }
        }

    } catch (err) {
        console.error('Math explainer error:', err)
        error.value = err.message || String(err)
    }
}
</script>

<style scoped>
.math-explainer {
    font-family: sans-serif;
    padding: 2rem;
    max-width: 600px;
    text-align: left;
}

input {
    width: 100%;
    padding: 0.5rem;
    margin-bottom: 1rem;
    font-size: 1rem;
    font-family: monospace;
}

button {
    padding: 0.5rem 1rem;
    font-size: 1rem;
}

ul {
    padding-left: 1.2rem;
}

li {
    margin-bottom: 0.4rem;
}
</style>