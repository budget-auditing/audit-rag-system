<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useQueryStore } from '@/store/query'
import { listDocuments } from '@/api/documents'
import type { EvidenceSource, DocumentOut } from '@/api/types'
import {
  MODE_OPTIONS,
  ROUTE_TO_MODE_CN,
  modeCn,
  methodName,
  routeTagType,
  conflictTagType,
  VERIFICATION_LABELS,
} from '@/utils/labels'

const store = useQueryStore()
const query = ref('')
const docScope = ref<string[]>([])
const documents = ref<DocumentOut[]>([])
// 用户主导选择的模式（route 取值）——默认信息查询
const selectedMode = ref('MSG_RAG')

onMounted(async () => {
  try {
    documents.value = await listDocuments()
  } catch {
    documents.value = []
  }
})
const traceDialogVisible = ref(false)
const traceSource = ref<EvidenceSource | null>(null)
const traceLoading = ref(false)
const hintLoading = ref(false)

async function fetchHint() {
  if (!query.value.trim()) return
  hintLoading.value = true
  try {
    await store.fetchModeHint(query.value)
  } finally {
    hintLoading.value = false
  }
}

function adoptSuggestion() {
  if (store.hint) selectedMode.value = store.hint.suggested_mode
}

async function submit() {
  const scope = docScope.value.length ? docScope.value : undefined
  // 用户所选模式为准，不做自动分类
  await store.ask({ query: query.value, doc_scope: scope, selected_mode: selectedMode.value })
}

async function openTrace(evidenceId: string) {
  traceDialogVisible.value = true
  traceLoading.value = true
  try {
    traceSource.value = await store.fetchEvidenceSource(evidenceId)
  } finally {
    traceLoading.value = false
  }
}

function conflictLabel(conflict: any) {
  const t = conflict.type || 'CONFLICT'
  if (conflict.values) return `${t}: ${conflict.values.join(' / ')}`
  if (conflict.message) return `${t}: ${conflict.message}`
  return t
}

function cleanAnswer(text: any): string {
  const s = typeof text === 'string' ? text : ''
  const i = s.lastIndexOf('</think>')
  if (i !== -1) {
    const tail = s.slice(i + '</think>'.length).trim()
    if (tail) return tail
  }
  return s.trim()
}

function allCaliberMetas(evidence: any[]) {
  const s = new Set<string>()
  for (const e of evidence) {
    const meta = e.caliber_meta || []
    if (Array.isArray(meta)) {
      meta.forEach((m: string) => s.add(m))
    }
  }
  return Array.from(s)
}

function verifyType(v: any): 'success' | 'danger' | 'info' {
  if (v === true) return 'success'
  if (v === false) return 'danger'
  return 'info'
}

function verifyText(v: any): string {
  if (v === true) return '是'
  if (v === false) return '否'
  return '未知'
}
</script>

<template>
  <div>
    <h2>审计问答</h2>

    <!-- 用户主导选择模式 -->
    <el-radio-group v-model="selectedMode" style="margin-bottom: 0.75rem; display: flex; gap: 0.75rem">
      <el-radio-button v-for="m in MODE_OPTIONS" :key="m.route" :value="m.route">
        {{ m.mode }}（{{ m.method }}）
      </el-radio-button>
    </el-radio-group>
    <p style="color: #909399; font-size: 0.85rem; margin: 0 0 0.75rem">
      {{ MODE_OPTIONS.find((m) => m.route === selectedMode)?.desc }}
    </p>

    <el-input
      v-model="query"
      type="textarea"
      :rows="3"
      placeholder="输入审计问题，例如：2024年某部门项目支出预算是多少？"
    />
    <el-select
      v-model="docScope"
      multiple
      filterable
      clearable
      placeholder="文档范围（可选，不选则检索全部）"
      style="margin-top: 0.5rem; width: 100%; display: block"
    >
      <el-option
        v-for="doc in documents"
        :key="doc.doc_id"
        :label="doc.doc_name"
        :value="doc.doc_id"
      />
    </el-select>

    <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem">
      <el-button :loading="hintLoading" @click="fetchHint">获取模式建议</el-button>
      <el-button type="primary" :loading="store.loading" @click="submit">开始查询</el-button>
    </div>

    <!-- 规则提示条（只建议，不自动执行） -->
    <el-alert
      v-if="store.hint"
      :closable="false"
      type="info"
      style="margin-top: 0.75rem"
    >
      <div>
        系统建议：<strong>{{ ROUTE_TO_MODE_CN[store.hint.suggested_mode] || store.hint.suggested_mode }}</strong>
      </div>
      <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem">
        <el-button size="small" @click="store.hint = null">保持当前</el-button>
        <el-button
          size="small"
          type="primary"
          :disabled="selectedMode === store.hint.suggested_mode"
          @click="adoptSuggestion"
        >
          切换为建议
        </el-button>
      </div>
    </el-alert>

    <div v-if="store.result" style="margin-top: 1.5rem">
      <el-card>
        <template #header>
          <div style="font-weight: bold; display: flex; align-items: center; gap: 0.5rem">
            <el-tag :type="routeTagType(store.result.route)">{{ methodName(store.result.route) }}</el-tag>
            <span>模式: {{ modeCn(store.result.route) }}</span>
          </div>
        </template>

        <!-- 模式与方法信息区 -->
        <el-descriptions :column="3" size="small" border style="margin-bottom: 1rem">
          <el-descriptions-item label="模式">{{ modeCn(store.result.route) }}</el-descriptions-item>
          <el-descriptions-item label="方法">{{ methodName(store.result.route) }}</el-descriptions-item>
          <el-descriptions-item v-if="store.result.metadata?.model_version" label="模型版本">
            {{ store.result.metadata.model_version }}
          </el-descriptions-item>
          <el-descriptions-item v-if="store.result.metadata?.index_version" label="索引版本">
            {{ store.result.metadata.index_version }}
          </el-descriptions-item>
          <el-descriptions-item v-if="store.result.metadata?.time_cost_sec" label="耗时(s)">
            {{ store.result.metadata.time_cost_sec }}
          </el-descriptions-item>
        </el-descriptions>

        <!-- 信息查询 (MSG-RAG) -->
        <div v-if="store.result.route === 'MSG_RAG' && store.result.answer.facts">
          <div v-if="store.result.answer.generated_answer" style="margin-bottom: 1rem">
            <h4>生成答案</h4>
            <el-alert type="success" :closable="false">
              <span style="white-space: pre-wrap">{{ cleanAnswer(store.result.answer.generated_answer) }}</span>
            </el-alert>
          </div>
          <h4>事实提取</h4>
          <el-table :data="store.result.answer.facts" border size="small">
            <el-table-column prop="subject" label="主体" />
            <el-table-column prop="predicate" label="谓词" />
            <el-table-column prop="object" label="对象" />
            <el-table-column prop="value" label="数值" />
            <el-table-column prop="unit" label="单位" />
          </el-table>
          <div v-if="store.result.answer.summary" style="margin-top: 0.5rem">
            <strong>摘要：</strong>{{ store.result.answer.summary }}
          </div>
        </div>

        <!-- 结论核验 (C³-RAG) -->
        <div v-else-if="store.result.route === 'C3_RAG'">
          <div v-if="store.result.answer.generated_answer" style="margin-bottom: 1rem">
            <h4>生成答案</h4>
            <el-alert type="success" :closable="false">
              <span style="white-space: pre-wrap">{{ cleanAnswer(store.result.answer.generated_answer) }}</span>
            </el-alert>
          </div>
          <div
            v-if="store.result.answer.consistency_score !== undefined || store.result.answer.consistency_result"
            style="margin-bottom: 0.5rem"
          >
            <h4>一致性核验</h4>
            <el-space wrap>
              <el-tag v-if="store.result.answer.consistency_score !== undefined" type="info">
                一致性评分: {{ store.result.answer.consistency_score }}
              </el-tag>
              <el-tag :type="store.result.answer.consistency_result?.caliber_consistent ? 'success' : 'danger'">
                口径一致: {{ store.result.answer.consistency_result?.caliber_consistent ? '是' : '否' }}
              </el-tag>
              <el-tag :type="store.result.answer.consistency_result?.causal_complete ? 'success' : 'warning'">
                因果完整: {{ store.result.answer.consistency_result?.causal_complete ? '是' : '否' }}
              </el-tag>
            </el-space>
          </div>
          <div v-if="store.result.answer.conflicts?.length" style="margin-top: 0.5rem">
            <h4>冲突标记</h4>
            <el-tag
              v-for="(f, i) in store.result.answer.conflicts"
              :key="i"
              :type="conflictTagType(f.type)"
              style="margin-right: 0.5rem; margin-bottom: 0.25rem"
            >
              {{ conflictLabel(f) }}
            </el-tag>
          </div>
          <div v-if="store.result.answer.evidence?.length" style="margin-top: 0.5rem">
            <h4>证据口径注记</h4>
            <el-tag
              v-for="(m, i) in allCaliberMetas(store.result.answer.evidence)"
              :key="i"
              size="small"
              style="margin-right: 0.5rem; margin-bottom: 0.25rem"
            >
              {{ m }}
            </el-tag>
          </div>
          <div v-if="store.result.answer.summary" style="margin-top: 0.5rem">
            <strong>摘要：</strong>{{ store.result.answer.summary }}
          </div>
        </div>

        <!-- 风险敏感片段索引 (TRACE) — OCI 中文三段 -->
        <div v-else-if="store.result.route === 'TRACE' || store.result.route === 'HIDDEN_RISK'">
          <div v-if="store.result.answer.observation || store.result.answer.cause || store.result.answer.impact">
            <h4>风险回答（OCI）</h4>
            <div v-if="store.result.answer.observation" style="margin-bottom: 0.5rem">
              <el-tag size="small" type="info">风险现象</el-tag>
              <p style="white-space: pre-wrap">{{ cleanAnswer(store.result.answer.observation) }}</p>
            </div>
            <div v-if="store.result.answer.cause" style="margin-bottom: 0.5rem">
              <el-tag size="small" type="warning">形成原因</el-tag>
              <p style="white-space: pre-wrap">{{ cleanAnswer(store.result.answer.cause) }}</p>
            </div>
            <div v-if="store.result.answer.impact" style="margin-bottom: 0.5rem">
              <el-tag size="small" type="danger">影响后果</el-tag>
              <p style="white-space: pre-wrap">{{ cleanAnswer(store.result.answer.impact) }}</p>
            </div>
          </div>
          <div
            v-else-if="store.result.answer.generated_answer"
            style="margin-bottom: 1rem"
          >
            <h4>生成答案</h4>
            <el-alert type="success" :closable="false">
              <span style="white-space: pre-wrap">{{ cleanAnswer(store.result.answer.generated_answer) }}</span>
            </el-alert>
          </div>

          <!-- 核验字段徽章 -->
          <div v-if="store.result.answer.verification" style="margin-top: 0.5rem">
            <h4>可验证性核验</h4>
            <el-space wrap>
              <el-tag
                v-for="(label, key) in VERIFICATION_LABELS"
                :key="key"
                :type="key === 'abstain'
                  ? (store.result.answer.verification[key] ? 'danger' : 'success')
                  : verifyType(store.result.answer.verification[key])"
                style="margin-bottom: 0.25rem"
              >
                {{ label }}: {{ verifyText(store.result.answer.verification[key]) }}
              </el-tag>
            </el-space>
          </div>

          <div v-if="store.result.answer.abstain" style="margin-top: 0.5rem">
            <el-alert title="系统弃权：证据不足" type="warning" :closable="false" />
          </div>
          <div v-if="store.result.answer.summary" style="margin-top: 0.5rem">
            <strong>摘要：</strong>{{ store.result.answer.summary }}
          </div>
        </div>

        <div v-else>
          <pre style="white-space: pre-wrap">{{ JSON.stringify(store.result.answer, null, 2) }}</pre>
        </div>

        <div v-if="store.result.warnings.length" style="margin-top: 1rem">
          <el-alert
            v-for="w in store.result.warnings"
            :key="w"
            :title="w"
            type="warning"
            :closable="false"
            style="margin-bottom: 0.5rem"
          />
        </div>

        <div v-if="store.result.evidence.length" style="margin-top: 1rem">
          <h4>证据展示</h4>
          <el-table :data="store.result.evidence" border size="small">
            <el-table-column type="index" width="50" />
            <el-table-column prop="chunk_id" label="chunk_id" width="140" />
            <el-table-column prop="text" label="文本" show-overflow-tooltip />
            <el-table-column label="操作" width="100">
              <template #default="{ row }">
                <el-button size="small" @click="openTrace(row.chunk_id)">溯源</el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-card>
    </div>

    <el-dialog v-model="traceDialogVisible" title="证据溯源" width="600px">
      <div v-if="traceLoading" v-loading="traceLoading" style="min-height: 120px" />
      <div v-else-if="traceSource">
        <p><strong>chunk_id:</strong> {{ traceSource.chunk_id }}</p>
        <p><strong>doc_id:</strong> {{ traceSource.doc_id }}</p>
        <el-input
          v-model="traceSource.text"
          type="textarea"
          :rows="6"
          readonly
        />
        <pre v-if="traceSource.source_location" style="margin-top: 0.5rem; background: #f8f9fa; padding: 0.5rem">{{ JSON.stringify(traceSource.source_location, null, 2) }}</pre>
      </div>
    </el-dialog>
  </div>
</template>
