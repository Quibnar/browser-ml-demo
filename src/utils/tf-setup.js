import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgpu'

let initialized = false

export async function initTF() {
  if (!initialized) {
    try {
      await tf.setBackend('webgpu')
    } catch (err) {
      console.warn('⚠️ WebGPU backend failed. Falling back to WebGL.')
      await tf.setBackend('webgl')
    }
    await tf.ready()
    console.log('✅ TF backend:', tf.getBackend())
    initialized = true
  }
  return tf
}
