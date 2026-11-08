import api from './client'
import type { EvidenceResponse, EvidenceSource } from './types'

export async function getEvidence(queryId: string): Promise<EvidenceResponse> {
  const res = await api.get<EvidenceResponse>(`/evidence/${queryId}`)
  return res.data
}

export async function getEvidenceSource(evidenceId: string): Promise<EvidenceSource> {
  const res = await api.get<EvidenceSource>(`/evidence/source/${evidenceId}`)
  return res.data
}
