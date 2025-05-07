<template>
    <div class="welcome">
      <h1>🔠 AI for Web Apps</h1>
      <h2>By Dirk Stockton</h2>
      <p>
        This demo suite showcases lightweight machine learning models that run entirely in the browser — no servers, no user data sent to the cloud, and no subscriptions required. Each module demonstrates a specific capability of in-browser ML using open-source libraries like TensorFlow.js, ONNX Runtime, and Transformers.js.
      </p>
  
      <h2>🚀 How to Use</h2>
      <p>Use the dropdown menu in the top-left corner to select a demo module. Each tool focuses on a different ML domain:</p>
      <ul>
        <li><strong>Behavior</strong> — Predicts mouse motion over time with real-time training.</li>
        <li><strong>Classification</strong> — Uses a local ONNX model to identify image contents.</li>
        <li><strong>Summarization</strong> — Groups pixel colors to simplify and represent images.</li>
        <li><strong>Analysis</strong> — Detects sentiment in short text using browser-trained embeddings.</li>
        <li><strong>Relevance</strong> — Performs semantic similarity ranking with Transformers.js.</li>
        <li><strong>Utility</strong> — Explains math expressions using symbolic algebra engines.</li>
      </ul>
  
      <h2>🛠️ Demo Code</h2>
      <p>
        All modules are implemented using <strong>Vite + Vue 3</strong> and rely on client-side ML libraries. The full source code is open-source and available on GitHub. You can fork, modify, or repurpose individual modules for your own projects — whether you're building education tools, real-time interfaces, or privacy-focused ML apps.
      </p>
      <p>
        🔎 Visit the repo: <a href="https://github.com/your-org/browser-ml-demo" target="_blank">github.com/your-org/browser-ml-demo</a>
      </p>
    </div>
  </template>
  
  <script setup>
  // No logic needed
  </script>
  
  <style scoped>
  .welcome {
    padding: 2rem;
    max-width: 750px;
    margin: 4rem auto;
    font-family: system-ui, sans-serif;
    line-height: 1.6;
    text-align: left;
    background-color: #111;
  }
  
  .welcome h1 {
    font-size: 2rem;
    margin-bottom: 1rem;
  }
  
  .welcome h2 {
    font-size: 1.25rem;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
  }
  
  .welcome a {
    color: #007acc;
    text-decoration: none;
  }
  
  .welcome a:hover {
    text-decoration: underline;
  }
  </style>
  