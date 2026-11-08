import { defineStore } from 'pinia'
import { ref } from 'vue'
import type {
  UnifiedAnswer,
  QueryRequest,
  QueryReportResponse,
  AuditDraftResponse,
  ModeHintResponse,
  EvidenceListResponse,
} from '@/api/types'
import * as queryApi from '@/api/query'
import * as evidenceApi from '@/api/evidence'
import * as exportApi from '@/api/export'

export const useQueryStore = defineStore('query', () => {
  const result = ref<UnifiedAnswer | null>(null)
  const loading = ref(false)
  const history = ref<UnifiedAnswer[]>([])
  const selectedLog = ref<UnifiedAnswer | null>(null)
  const hint = ref<ModeHintResponse | null>(null)

  async function ask(payload: QueryRequest) {
    loading.value = true
    try {
      result.value = await queryApi.answerQuery(payload)
    } finally {
      loading.value = false
    }
  }

  async function fetchModeHint(query: string) {
    hint.value = await queryApi.modeHint(query)
    return hint.value
  }

  async function exportEvidenceList(queryIds: string[]): Promise<EvidenceListResponse> {
    return exportApi.exportEvidenceList({ query_ids: queryIds, format: 'markdown' })
  }

  async function loadHistory() {
    history.value = await queryApi.getQueryHistory()
  }

  async function fetchEvidence(queryId: string) {
    return evidenceApi.getEvidence(queryId)
  }

  async function fetchEvidenceSource(evidenceId: string) {
    return evidenceApi.getEvidenceSource(evidenceId)
  }

  async function exportReport(queryId: string): Promise<QueryReportResponse> {
    return exportApi.exportQueryReport({ query_id: queryId, format: 'markdown' })
  }

  async function exportAuditDraft(queryIds: string[]): Promise<AuditDraftResponse> {
    return exportApi.exportAuditDraft({ query_ids: queryIds, format: 'markdown' })
  }

  function selectLog(log: UnifiedAnswer | null) {
    selectedLog.value = log
  }

  return {
    result, loading, history, selectedLog, hint,
    ask, fetchModeHint, loadHistory, fetchEvidence, fetchEvidenceSource,
    exportReport, exportAuditDraft, exportEvidenceList, selectLog,
  }
})
