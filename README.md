# 🧠 Browser ML Demos

A lightweight showcase of in-browser machine learning tools built with **Vue.js**, **TensorFlow.js**, and **Transformers.js**. No paid APIs, no backend servers — just pure WebML and open-source models.

---

## 🚀 Live Demos

Each module runs entirely in your browser using JavaScript, WebGPU, or WASM. Select a demo from the dropdown menu to try it live.

---

## 🧩 Included Modules

### 🖱️ Behavior — *MouseTrainer.vue*
- Uses TensorFlow.js + WebGPU to predict cursor movement in real time.
- Model trains while you move the mouse.
- Applications: gesture recognition, stylus prediction, motion smoothing.

### 🧠 Classification — *ImageClassifier.vue*
- Uses ONNX Runtime Web to classify images using MobileNetV2.
- Runs entirely with WASM; no cloud inference required.
- Applications: object detection, sorting, camera input filtering.

### 🎨 Summarization — *ColorClustering.vue*
- Applies k-means clustering to reduce image color complexity.
- Simple and fast pixel summarization.
- Applications: palette extraction, thumbnail compression, segmentation.

### 😃 Analysis — *SentimentAnalyzer.vue*
- Uses Transformers.js to analyze sentiment of user-provided text.
- Browser-based BERT model (quantized for speed).
- Applications: review analysis, social tools, feedback sorting.

### 🔍 Relevance — *VectorSearch.vue*
- Embeds a query and candidates into semantic vectors (MiniLM).
- Compares using cosine similarity and ranks results.
- Applications: FAQ search, document retrieval, chatbot memory.

### 📐 Utility — *MathExplainer.vue*
- Uses `mathjs` and `nerdamer` to parse and explain symbolic math expressions.
- Shows step-by-step simplifications and symbolic results.
- Applications: tutoring, equation debugging, symbolic calculators.

---

## 📦 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/browser-ml-demo.git
cd browser-ml-demo
```

## 🔧 Tech Stack

- [Vite.js](https://vitejs.dev/)
- [Vue 3](https://vuejs.org/)
- [TensorFlow.js](https://www.tensorflow.org/js)
- [ONNX Runtime Web](https://onnxruntime.ai/)
- [Transformers.js](https://xenova.github.io/transformers.js/)
- [math.js](https://mathjs.org/)
- [nerdamer](https://nerdamer.com/)

## 📱 Device Support

This demo is currently designed for **desktop devices** only. Mobile and tablet support is disabled to avoid misleading UX and performance issues. A mobile-optimized version may be released in the future.

## 📚 Learn More

Browser-native ML tools can be useful alternatives to large cloud APIs like OpenAI. Many libraries support offline tasks such as:

- **Symbolic computation** (`mathjs`, `nerdamer`)
- **Embeddings & semantic search** (`transformers.js`)
- **Object detection & vision models** (`onnxruntime-web`)
- **Signal processing & clustering** (`scikit.js`, `ml5`)
- **Time-series learning** (`tfjs`)

Explore the code, mix the modules, and use them in your own app — it's all open source.
