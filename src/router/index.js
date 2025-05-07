
import { createRouter, createWebHistory } from 'vue-router'
import App from '../App.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: App, // We'll use dropdown logic inside App.vue for now
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
