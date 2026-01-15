<template>
  <div class="welcome-container">
    <div class="welcome-content">
      <h1>🔠 AI for Web Apps</h1>
      <h2>By Dirk Stockton</h2>
      <p class="intro-text">
        This comprehensive demo suite showcases cutting-edge machine learning capabilities that run entirely in your browser — no servers, no user data sent to the cloud, and no subscriptions required. From real-time training and interactive drawing to advanced computer vision and natural language processing, each module demonstrates the power of in-browser ML using open-source libraries like TensorFlow.js, ONNX Runtime, and Transformers.js.
      </p>

      <h2>🚀 How to Use</h2>
      <p>Use the navigation menu on the left to select a demo module. Each tool focuses on a different ML domain:</p>
      <ul class="demo-list">
        <li><strong>Behavior Modeling</strong> — Predicts mouse motion over time with real-time training, heat maps, and flow fields.</li>
        <li><strong>Image Classification</strong> — Uses a local ONNX model to identify image contents with high accuracy.</li>
        <li><strong>Color Clustering</strong> — Groups pixel colors to simplify and represent images using K-means clustering.</li>
        <li><strong>Sentiment Analysis</strong> — Detects sentiment in short text using browser-trained embeddings.</li>
        <li><strong>Vector Search</strong> — Performs semantic similarity ranking with Transformers.js for intelligent search.</li>
        <li><strong>Math Explainer</strong> — Explains math expressions using symbolic algebra engines (Math.js & Nerdamer).</li>
        <li><strong>Batch Training</strong> — Visualizes real-time CNN training on MNIST and Fashion MNIST datasets.</li>
        <li><strong>Live Digit Trainer</strong> — Interactive drawing and training of custom digit recognition models.</li>
      </ul>

      <h2>🌟 Key Features</h2>
      <ul class="feature-list">
        <li><strong>Privacy-First:</strong> All ML processing happens locally in your browser — no data sent to servers</li>
        <li><strong>Real-Time Learning:</strong> Watch models train and adapt in real-time with live visualizations</li>
        <li><strong>Hardware Acceleration:</strong> Automatic WebGPU → WebGL → CPU fallback for optimal performance</li>
        <li><strong>Interactive Experience:</strong> Draw, train, and test models with immediate feedback</li>
        <li><strong>Production Ready:</strong> Built with industry-standard libraries and best practices</li>
        <li><strong>Educational:</strong> Perfect for learning ML concepts through hands-on experimentation</li>
      </ul>

      <h2>🛠️ Demo Code</h2>
      <p>
        All modules are implemented using <strong>Vite + Vue 3</strong> and rely on client-side ML libraries. The full source code is open-source and available on GitHub. You can fork, modify, or repurpose individual modules for your own projects — whether you're building education tools, real-time interfaces, or privacy-focused ML apps.
      </p>
      <p class="repo-link">
        🔎 Visit the repo: <a href="https://github.com/your-org/browser-ml-demo" target="_blank" rel="noopener">github.com/your-org/browser-ml-demo</a>
      </p>
    </div>
  </div>
</template>

<script setup>
// No logic needed
</script>

<style scoped>
.welcome-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 80vh;
  padding: 2rem;
}

.welcome-content {
  max-width: 800px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 3rem;
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  line-height: 1.6;
  text-align: left;
}

.welcome-content h1 {
  font-size: 3rem;
  margin-bottom: 1rem;
  color: #333;
  text-align: center;
}

.welcome-content h2 {
  font-size: 1.5rem;
  margin-top: 2.5rem;
  margin-bottom: 1rem;
  color: #555;
  border-bottom: 2px solid #007bff;
  padding-bottom: 0.5rem;
}

.intro-text {
  font-size: 1.1rem;
  color: #444;
  text-align: center;
  font-style: italic;
  margin-bottom: 2rem;
}

.demo-list {
  padding-left: 1.5rem;
  margin: 1rem 0;
}

.demo-list li {
  margin-bottom: 0.75rem;
  color: #333;
}

.demo-list strong {
  color: #007bff;
}

.feature-list {
  padding-left: 1.5rem;
  margin: 1rem 0;
}

.feature-list li {
  margin-bottom: 0.75rem;
  color: #333;
}

.feature-list strong {
  color: #28a745;
}

.repo-link {
  text-align: center;
  margin-top: 2rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}

.welcome-content p {
  color: #333;
  text-decoration: none;
  font-weight: 600;
  transition: color 0.2s;
  text-align: left;
}

.welcome-content a {
  color: #007bff;
  text-decoration: none;
  font-weight: 600;
  transition: color 0.2s;
}

.welcome-content a:hover {
  color: #0056b3;
  text-decoration: underline;
}

@media (max-width: 768px) {
  .welcome-container {
    padding: 1rem;
  }
  
  .welcome-content {
    padding: 2rem;
  }
  
  .welcome-content h1 {
    font-size: 2rem;
  }
  
  .welcome-content h2 {
    font-size: 1.3rem;
  }
}
</style>
  