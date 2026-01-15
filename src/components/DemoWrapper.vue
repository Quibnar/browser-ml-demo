<template>
  <div class="demo-wrapper">
    <!-- Tool area -->
    <div class="tool-section">
      <slot name="tool"/>
    </div>

    <!-- Explanation/sidebar area -->
    <div class="explanation-section">
      <slot name="explanation"/>
    </div>
  </div>
</template>

<script setup>
// No props needed—everything is passed via slots
</script>

<style scoped>
.demo-wrapper {
  width: 100%;
  max-width: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  align-items: start;
  min-height: calc(100vh - 40px); /* Account for padding */
}

.tool-section {
  padding: 2rem;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid #e1e5e9;
  min-height: 400px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.explanation-section {
  padding: 2rem;
  background: #f8f9fa;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid #e1e5e9;
  color: #333;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
}

.explanation-section h1 {
  color: #2c3e50;
  margin-bottom: 1rem;
  font-size: 2rem;
  font-weight: 600;
}

.explanation-section h2 {
  color: #34495e;
  margin: 1.5rem 0 1rem 0;
  font-size: 1.5rem;
  font-weight: 500;
}

.explanation-section h3 {
  color: #7f8c8d;
  margin: 1.5rem 0 0.5rem 0;
  font-size: 1.2rem;
  font-weight: 500;
}

.explanation-section p {
  line-height: 1.6;
  margin-bottom: 1rem;
  color: #555;
}

.explanation-section ul {
  margin: 1rem 0;
  padding-left: 1.5rem;
}

.explanation-section li {
  margin-bottom: 0.5rem;
  line-height: 1.5;
  color: #555;
}

.explanation-section strong {
  color: #2c3e50;
  font-weight: 600;
}

/* Loading and error states */
.loading {
  font-size: 1.2rem;
  text-align: center;
  padding: 2rem;
  color: #666;
}

.error {
  color: #e74c3c;
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(231, 76, 60, 0.1);
  border-radius: 8px;
  border: 1px solid rgba(231, 76, 60, 0.3);
}

/* Responsive design */
@media (max-width: 1024px) {
  .demo-wrapper {
    grid-template-columns: 1fr;
    gap: 2rem;
    padding: 0;
  }
  
  .explanation-section {
    max-height: none;
    overflow-y: visible;
  }
}

@media (max-width: 768px) {
  .demo-wrapper {
    margin: 0;
    padding: 0;
    gap: 1rem;
  }
  
  .tool-section,
  .explanation-section {
    padding: 1.5rem;
    border-radius: 8px;
  }
  
  .explanation-section h1 {
    font-size: 1.5rem;
  }
  
  .explanation-section h2 {
    font-size: 1.3rem;
  }
  
  .explanation-section h3 {
    font-size: 1.1rem;
  }
}

/* Custom scrollbar for explanation section */
.explanation-section::-webkit-scrollbar {
  width: 8px;
}

.explanation-section::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.05);
  border-radius: 4px;
}

.explanation-section::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}

.explanation-section::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.3);
}
</style>
