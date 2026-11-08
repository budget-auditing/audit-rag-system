import api from './client'
import type { OverviewResponse, SystemStatusResponse } from './types'

export async function getOverview(): Promise<OverviewResponse> {
  const res = await api.get('/stats/overview')
  return res.data
}

export async function getSystemStatus(): Promise<SystemStatusResponse> {
  const res = await api.get('/system/status')
  return res.data
}
