import api from './client'
import type {
  QueryReportRequest,
  QueryReportResponse,
  AuditDraftRequest,
  AuditDraftResponse,
  EvidenceListRequest,
  EvidenceListResponse,
} from './types'

export async function exportEvidenceList(payload: EvidenceListRequest): Promise<EvidenceListResponse> {
  const res = await api.post<EvidenceListResponse>(`/export/evidence-list`, payload)
  return res.data
}

export async function exportQueryReport(payload: QueryReportRequest): Promise<QueryReportResponse> {
  const res = await api.post<QueryReportResponse>(`/export/query-report`, payload)
  return res.data
}

export async function exportAuditDraft(payload: AuditDraftRequest): Promise<AuditDraftResponse> {
  const res = await api.post<AuditDraftResponse>(`/export/audit-draft`, payload)
  return res.data
}
