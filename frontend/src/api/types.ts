export interface UnifiedAnswer {
  query_id: string
  query: string
  query_type: string
  route: string
  confidence: number
  answer: Record<string, any>
  evidence: any[]
  warnings: string[]
  metadata: Record<string, any>
}

export interface DocumentOut {
  doc_id: string
  doc_name: string
  doc_type?: string
  year?: string
  department?: string
  parse_status?: string
  index_status?: string
}

export interface HealthResponse {
  status: string
  service: string
}

export interface QueryRequest {
  query: string
  doc_scope?: string[]
  force_route?: string | null
  selected_mode?: string | null
}

export interface ModeHintResponse {
  suggested_mode: string
  scores: Record<string, number>
  reason: string
}

export interface RecentTask {
  task_id: string
  task_type: string
  task_type_cn: string
  target: string
  status: string
  detail: string
  created_at: string
  finished_at: string | null
}

export interface OverviewResponse {
  documents: { total: number; parsed: number }
  indexes: Record<string, any>
  queries: { total: number }
  recent_tasks: RecentTask[]
  recent_jobs: any[]
}

export interface SystemStatusResponse {
  components: Record<string, any>
}

export interface EvidenceListRequest {
  query_ids: string[]
  format?: string
}

export interface EvidenceListResponse {
  content: string
  format: string
}

export interface BuildJob {
  job_id: string
  target: string
  status: 'queued' | 'building' | 'ready' | 'failed'
  phase: string
  progress: { current: number; total: number }
  message?: string
  result?: Record<string, any> | null
  error?: string | null
  created_at?: string
  started_at?: string | null
  finished_at?: string | null
}

export interface BuildIndexResponse {
  built: boolean
  queued?: boolean
  message?: string
  job?: BuildJob | null
}

export interface BuildStatusResponse {
  job: BuildJob | null
  active: boolean
}

export interface IndexStatus {
  c3rag: Record<string, any>
  trace: Record<string, any>
  msg: Record<string, any>
  corpus?: Record<string, any>
  last_update?: string
}

export interface EvidenceSource {
  chunk_id: string
  doc_id: string
  text: string
  source_location?: Record<string, any>
}

export interface EvidenceResponse {
  query_id: string
  query: string
  route: string
  confidence: number
  answer: Record<string, any>
  evidence: any[]
  warnings: string[]
  metadata: Record<string, any>
}

export interface QueryReportRequest {
  query_id: string
  format?: string
}

export interface QueryReportResponse {
  content: string
  format: string
}

export interface AuditDraftRequest {
  query_ids: string[]
  format?: string
}

export interface AuditDraftResponse {
  content: string
  format: string
}
