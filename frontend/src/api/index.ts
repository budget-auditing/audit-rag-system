import api from './client'
import type { IndexStatus, BuildIndexResponse, BuildStatusResponse } from './types'

export async function getIndexStatus(): Promise<IndexStatus> {
  const res = await api.get('/index/status')
  return res.data
}

export async function getBuildStatus(jobId?: string): Promise<BuildStatusResponse> {
  const res = await api.get('/index/build/status', { params: jobId ? { job_id: jobId } : {} })
  return res.data
}

export async function buildAllIndexes(): Promise<BuildIndexResponse> {
  const res = await api.post('/index/build/all')
  return res.data
}

export async function buildFactGraph(): Promise<BuildIndexResponse> {
  const res = await api.post('/index/build/fact-graph')
  return res.data
}

export async function buildInstitutionalIndex(): Promise<BuildIndexResponse> {
  const res = await api.post('/index/build/institutional')
  return res.data
}

export async function buildRiskIndex(): Promise<BuildIndexResponse> {
  const res = await api.post('/index/build/risk')
  return res.data
}
