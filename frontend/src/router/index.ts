import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/dashboard' },
  { path: '/dashboard', name: 'Dashboard', component: () => import('@/views/DashboardView.vue') },
  { path: '/documents', name: 'Documents', component: () => import('@/views/DocumentsView.vue') },
  { path: '/index', name: 'Index', component: () => import('@/views/IndexView.vue') },
  { path: '/qa', name: 'QA', component: () => import('@/views/QAView.vue') },
  { path: '/logs', name: 'Logs', component: () => import('@/views/LogsView.vue') },
  { path: '/status', name: 'SystemStatus', component: () => import('@/views/SystemStatusView.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
