import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '/api',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

export default api
