// 展示层术语映射。route 取值保持 MSG_RAG/C3_RAG/TRACE 不变，
// 仅在界面上呈现中文模式名与方法名，避免破坏 query_logs 与后端契约。

// route → 中文模式名
export const ROUTE_TO_MODE_CN: Record<string, string> = {
  MSG_RAG: '信息查询',
  C3_RAG: '结论核验',
  TRACE: '风险识别',
}

// route → 方法名
export const ROUTE_TO_METHOD: Record<string, string> = {
  MSG_RAG: 'MSG-RAG',
  C3_RAG: 'C³-RAG',
  TRACE: 'TRACE',
}

// qtype → 中文模式名（与 route 一一对应）
export const QTYPE_TO_MODE_CN: Record<string, string> = {
  fact_numeric: '信息查询',
  institutional_consistency: '结论核验',
  hidden_risk: '风险识别',
}

export function modeCn(route: string): string {
  return ROUTE_TO_MODE_CN[route] || route
}

export function methodName(route: string): string {
  return ROUTE_TO_METHOD[route] || route
}

export function routeTagType(route: string): 'success' | 'warning' | 'danger' | 'info' {
  if (route === 'MSG_RAG') return 'success'
  if (route === 'C3_RAG') return 'warning'
  if (route === 'TRACE' || route === 'HIDDEN_RISK') return 'danger'
  return 'info'
}

// 三类可主动选择的模式入口
export const MODE_OPTIONS = [
  { route: 'MSG_RAG', mode: '信息查询', method: 'MSG-RAG', desc: '事实与数值查询，如“某项目预算金额是多少”。' },
  { route: 'C3_RAG', mode: '结论核验', method: 'C³-RAG', desc: '口径可比、因果完整性核验。' },
  { route: 'TRACE', mode: '风险识别', method: 'TRACE', desc: '可持续性风险、执行偏差，OCI 结构化推理。' },
]

// 冲突标签样式
export function conflictTagType(type: string): 'success' | 'warning' | 'danger' | 'info' {
  if (type === 'YEAR_MISMATCH' || type === 'DEPT_MISMATCH') return 'warning'
  if (type === 'CALIBER_MISMATCH' || type === 'METRIC_MISMATCH') return 'danger'
  if (type === 'CAUSAL_GAP' || type === 'EVIDENCE_INSUFFICIENT') return 'info'
  return 'danger'
}

// 核验字段中文标签
export const VERIFICATION_LABELS: Record<string, string> = {
  claim_supported: '论断可支持',
  attribute_consistent: '属性一致',
  single_passage_supported: '单段可支撑',
  abstain: '弃权',
}
