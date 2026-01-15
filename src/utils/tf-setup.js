import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgpu'
import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-cpu'

let initialized = false
let currentBackend = null

export async function initTF() {
  if (!initialized) {
    try {
      // Try WebGPU first for best performance
      await tf.setBackend('webgpu')
      currentBackend = 'webgpu'
      console.log('✅ Using WebGPU backend')
      
      // WebGPU-specific optimizations (only if available)
      if (tf.getBackend() === 'webgpu') {
        try {
          // Only set flags that are registered and commonly available
          const webgpuFlags = [
            'WEBGPU_CPU_FORWARD',
            'WEBGPU_DEPTH_TEXTURE'
          ]
          
          webgpuFlags.forEach(flag => {
            try {
              if (tf.env().get(flag) !== undefined) {
                if (flag === 'WEBGPU_CPU_FORWARD') {
                  tf.env().set(flag, false)
                } else if (flag === 'WEBGPU_DEPTH_TEXTURE') {
                  tf.env().set(flag, false)
                }
              }
            } catch (flagErr) {
              // Silently ignore flag errors
            }
          })
          
          // Set memory management (only if available)
          const memoryFlags = [
            'WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE',
            'WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD'
          ]
          
          memoryFlags.forEach(flag => {
            try {
              if (tf.env().get(flag) !== undefined) {
                if (flag === 'WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE') {
                  tf.env().set(flag, 100)
                } else if (flag === 'WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD') {
                  tf.env().set(flag, 1000)
                }
              }
            } catch (flagErr) {
              // Silently ignore flag errors
            }
          })
          
          console.log('🚀 WebGPU backend active with optimizations')
        } catch (optErr) {
          console.log('🚀 WebGPU backend active (basic mode)')
        }
      }
    } catch (err) {
      console.warn('⚠️ WebGPU backend failed, trying WebGL...')
      try {
        await tf.setBackend('webgl')
        currentBackend = 'webgl'
        console.log('✅ Using WebGL backend')
        
        // WebGL-specific optimizations (only if available)
        if (tf.getBackend() === 'webgl') {
          try {
            const gl = tf.backend().gpgpu.gl
            if (gl) {
              // Only disable features that are valid capabilities
              const validCapabilities = [
                gl.DEPTH_TEST, gl.STENCIL_TEST, gl.BLEND, gl.DITHER,
                gl.POLYGON_OFFSET_FILL, gl.SAMPLE_COVERAGE_INVERT, gl.SAMPLE_ALPHA_TO_COVERAGE
              ]
              
              validCapabilities.forEach(cap => {
                try {
                  if (gl.isEnabled(cap)) {
                    gl.disable(cap)
                  }
                } catch (capErr) {
                  // Silently ignore invalid capability errors
                }
              })
              
              // Enable performance optimizations
              try {
                gl.enable(gl.CULL_FACE)
                gl.cullFace(gl.BACK)
              } catch (cullErr) {
                // Silently ignore culling errors
              }
            }
            
            // WebGL memory management (only if available)
            const webglFlags = [
              'WEBGL_FORCE_F16_TEXTURES',
              'WEBGL_PACK',
              'WEBGL_PACK_DEPTHWISECONV',
              'WEBGL_FLUSH_THRESHOLD'
            ]
            
            webglFlags.forEach(flag => {
              try {
                if (tf.env().get(flag) !== undefined) {
                  if (flag === 'WEBGL_FORCE_F16_TEXTURES') {
                    tf.env().set(flag, false)
                  } else if (flag === 'WEBGL_PACK') {
                    tf.env().set(flag, true)
                  } else if (flag === 'WEBGL_PACK_DEPTHWISECONV') {
                    tf.env().set(flag, true)
                  } else if (flag === 'WEBGL_FLUSH_THRESHOLD') {
                    tf.env().set(flag, 1)
                  }
                }
              } catch (flagErr) {
                // Silently ignore flag errors
              }
            })
            
            console.log('⚡ WebGL optimizations enabled')
          } catch (webglOptErr) {
            console.log('⚡ WebGL backend active (basic mode)')
          }
        }
      } catch (webglErr) {
        console.warn('⚠️ WebGL backend failed, falling back to CPU...')
        try {
          await tf.setBackend('cpu')
          currentBackend = 'cpu'
          console.log('✅ Using CPU backend')
          
          // CPU-specific optimizations (only if available)
          try {
            if (tf.env().get('CPU_HANDOFF_SIZE_THRESHOLD') !== undefined) {
              tf.env().set('CPU_HANDOFF_SIZE_THRESHOLD', 1000)
            }
            if (tf.env().get('WEBGL_CPU_FORWARD') !== undefined) {
              tf.env().set('WEBGL_CPU_FORWARD', false)
            }
            
            console.log('🐌 CPU backend (limited optimizations)')
          } catch (cpuOptErr) {
            console.log('🐌 CPU backend active (basic mode)')
          }
        } catch (cpuErr) {
          throw new Error('All TensorFlow backends failed to initialize')
        }
      }
    }
    
    await tf.ready()
    
    // Global TensorFlow optimizations (only if available)
    const globalFlags = [
      'WEBGL_DELETE_TEXTURE_THRESHOLD',
      'WEBGL_USE_ANGLE',
      'WEBGL_CPU_FORWARD',
      'WEBGL_FORCE_F16_TEXTURES',
      'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_REQUIRED',
      'WEBGL_DEPTH_TEXTURE',
      'WEBGL_VERSION',
      'WEBGL_FLUSH_THRESHOLD',
      'WEBGL_PACK',
      'WEBGL_PACK_DEPTHWISECONV'
    ]
    
    globalFlags.forEach(flag => {
      try {
        if (tf.env().get(flag) !== undefined) {
          if (flag === 'WEBGL_DELETE_TEXTURE_THRESHOLD') {
            tf.env().set(flag, 0)
          } else if (flag === 'WEBGL_USE_ANGLE') {
            tf.env().set(flag, false)
          } else if (flag === 'WEBGL_CPU_FORWARD') {
            tf.env().set(flag, false)
          } else if (flag === 'WEBGL_FORCE_F16_TEXTURES') {
            tf.env().set(flag, false)
          } else if (flag === 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_REQUIRED') {
            tf.env().set(flag, false)
          } else if (flag === 'WEBGL_DEPTH_TEXTURE') {
            tf.env().set(flag, false)
          } else if (flag === 'WEBGL_VERSION') {
            tf.env().set(flag, 1)
          } else if (flag === 'WEBGL_FLUSH_THRESHOLD') {
            tf.env().set(flag, 1)
          } else if (flag === 'WEBGL_PACK') {
            tf.env().set(flag, true)
          } else if (flag === 'WEBGL_PACK_DEPTHWISECONV') {
            tf.env().set(flag, true)
          }
        }
      } catch (flagErr) {
        // Silently ignore flag errors
      }
    })
    
    console.log('🎯 TensorFlow.js initialized with available optimizations')
    initialized = true
  }
  return tf
}

export function getCurrentBackend() {
  return currentBackend
}

export function isWebGPUAvailable() {
  return currentBackend === 'webgpu'
}

export function isWebGLAvailable() {
  return currentBackend === 'webgl'
}

export function isCPUAvailable() {
  return currentBackend === 'cpu'
}

// Performance monitoring utilities
export function getBackendPerformance() {
  try {
    const backend = tf.getBackend()
    const memoryInfo = tf.memory()
    
    return {
      backend,
      numTensors: memoryInfo.numTensors,
      numBytes: memoryInfo.numBytes,
      numDataBuffers: memoryInfo.numDataBuffers,
      gpuMemory: memoryInfo.gpuMemory
    }
  } catch (err) {
    console.warn('⚠️ Could not get performance info:', err.message)
    return { backend: 'unknown', error: err.message }
  }
}

export function optimizeForTraining() {
  try {
    // Optimize TensorFlow for training workloads (only if available)
    const trainingFlags = [
      'WEBGL_FORCE_F16_TEXTURES',
      'WEBGL_PACK',
      'WEBGL_PACK_DEPTHWISECONV',
      'WEBGL_FLUSH_THRESHOLD'
    ]
    
    trainingFlags.forEach(flag => {
      try {
        if (tf.env().get(flag) !== undefined) {
          if (flag === 'WEBGL_FORCE_F16_TEXTURES') {
            tf.env().set(flag, false)
          } else if (flag === 'WEBGL_PACK') {
            tf.env().set(flag, true)
          } else if (flag === 'WEBGL_PACK_DEPTHWISECONV') {
            tf.env().set(flag, true)
          } else if (flag === 'WEBGL_FLUSH_THRESHOLD') {
            tf.env().set(flag, 1)
          }
        }
      } catch (flagErr) {
        // Silently ignore flag errors
      }
    })
    
    console.log('🎯 Optimized for training workloads')
  } catch (err) {
    console.warn('⚠️ Training optimization failed:', err.message)
  }
}

export function optimizeForInference() {
  try {
    // Optimize TensorFlow for inference workloads (only if available)
    const inferenceFlags = [
      'WEBGL_FORCE_F16_TEXTURES',
      'WEBGL_PACK',
      'WEBGL_FLUSH_THRESHOLD'
    ]
    
    inferenceFlags.forEach(flag => {
      try {
        if (tf.env().get(flag) !== undefined) {
          if (flag === 'WEBGL_FORCE_F16_TEXTURES') {
            tf.env().set(flag, true)
          } else if (flag === 'WEBGL_PACK') {
            tf.env().set(flag, false)
          } else if (flag === 'WEBGL_FLUSH_THRESHOLD') {
            tf.env().set(flag, 100)
          }
        }
      } catch (flagErr) {
        // Silently ignore flag errors
      }
    })
    
    console.log('🎯 Optimized for inference workloads')
  } catch (err) {
    console.warn('⚠️ Inference optimization failed:', err.message)
  }
}
