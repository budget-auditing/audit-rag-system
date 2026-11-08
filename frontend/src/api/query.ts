import api from './client'
import type { UnifiedAnswer, QueryRequest, ModeHintResponse } from './types'

export async function modeHint(query: string): Promise<ModeHintResponse> {
  const res = await api.post('/qa/mode-hint', { query })
  return res.data
}

export async function answerQuery(payload: QueryRequest): Promise<UnifiedAnswer> {
  // LLM-backed routes (C³-RAG/TRACE/MSG) routinely take 30-300s; override the
  // shorter global timeout for this call.
  const res = await api.post('/query/answer', payload, { timeout: 300000 })
  return res.data
}

export async function getQueryHistory(): Promise<UnifiedAnswer[]> {
  const res = await api.get('/query/history')
  return res.data
}
