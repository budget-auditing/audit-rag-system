<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useQueryStore } from '@/store/query'
import type { UnifiedAnswer } from '@/api/types'
import { ElMessage } from 'element-plus'
import { modeCn, routeTagType } from '@/utils/labels'

const store = useQueryStore()
const detailDrawer = ref(false)
const selectedIds = ref<string[]>([])

onMounted(() => store.loadHistory())

function openDetail(row: UnifiedAnswer) {
  store.selectLog(row)
  detailDrawer.value = true
}

async function exportSingle(row: UnifiedAnswer) {
  try {
    const res = await store.exportReport(row.query_id)
    ElMessage.success('报告内容长度：' + (res.content?.length ?? 0))
  } catch (e) {
    ElMessage.error('导出失败：' + String(e))
  }
}

async function generateAuditDraft() {
  if (selectedIds.value.length === 0) {
    ElMessage.warning('请至少选择一条记录')
    return
  }
  try {
    const res = await store.exportAuditDraft(selectedIds.value)
    ElMessage.success('审计草稿内容长度：' + (res.content?.length ?? 0))
  } catch (e) {
    ElMessage.error('生成失败：' + String(e))
  }
}

async function exportEvidenceList() {
  if (selectedIds.value.length === 0) {
    ElMessage.warning('请至少选择一条记录')
    return
  }
  try {
    const res = await store.exportEvidenceList(selectedIds.value)
    ElMessage.success('证据清单内容长度：' + (res.content?.length ?? 0))
  } catch (e) {
    ElMessage.error('导出失败：' + String(e))
  }
}

function ruleHintText(row: UnifiedAnswer): string {
  const h = row.metadata?.rule_hint
  if (!h) return '-'
  const sm = h.suggested_mode ? (modeCn(h.suggested_mode)) : ''
  return sm ? `建议：${sm}` : '-'
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
</script>

<template>
  <div>
    <h2>问答日志</h2>
    <el-space style="margin-bottom: 1rem">
      <el-button @click="store.loadHistory()">刷新</el-button>
      <el-button type="primary" @click="generateAuditDraft">生成审计草稿</el-button>
      <el-button @click="exportEvidenceList">导出证据清单</el-button>
    </el-space>
    <el-table
      :data="store.history"
      border
      @selection-change="(rows: UnifiedAnswer[]) => { selectedIds = rows.map(r => r.query_id) }"
    >
      <el-table-column type="selection" width="55" />
      <el-table-column prop="query_id" label="ID" width="120" />
      <el-table-column prop="query" label="问题" min-width="320" />
      <el-table-column prop="query_type" label="类型" width="140" />
      <el-table-column label="路由" width="120">
        <template #default="{ row }">
          <el-tag :type="routeTagType(row.route)">{{ modeCn(row.route) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="所选模式" width="110">
        <template #default="{ row }">
          {{ row.metadata?.selected_mode ? modeCn(row.metadata.selected_mode) : '自动' }}
        </template>
      </el-table-column>
      <el-table-column label="规则提示" width="130">
        <template #default="{ row }">{{ ruleHintText(row) }}</template>
      </el-table-column>
      <el-table-column label="耗时(s)" width="90">
        <template #default="{ row }">{{ row.metadata?.time_cost_sec ?? '-' }}</template>
      </el-table-column>
      <el-table-column label="操作" width="180">
        <template #default="{ row }">
          <el-button size="small" @click="openDetail(row)">详情</el-button>
          <el-button size="small" @click="exportSingle(row)">导出</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-drawer v-model="detailDrawer" title="日志详情" size="600px">
      <div v-if="store.selectedLog">
        <p><strong>ID:</strong> {{ store.selectedLog.query_id }}</p>
        <p><strong>问题:</strong> {{ store.selectedLog.query }}</p>
        <p><strong>类型:</strong> {{ store.selectedLog.query_type }}</p>
        <p>
          <strong>路由:</strong>
          <el-tag :type="routeTagType(store.selectedLog.route)">{{ store.selectedLog.route }}</el-tag>
        </p>

        <h4>答案</h4>
        <el-alert
          v-if="cleanAnswer(store.selectedLog.answer?.generated_answer)"
          type="success"
          :closable="false"
          style="margin-bottom: 0.75rem"
        >
          <span style="white-space: pre-wrap">{{ cleanAnswer(store.selectedLog.answer.generated_answer) }}</span>
        </el-alert>

        <div
          v-if="store.selectedLog.answer?.observation || store.selectedLog.answer?.cause || store.selectedLog.answer?.impact"
          style="margin-bottom: 0.75rem"
        >
          <div v-if="store.selectedLog.answer.observation" style="margin-bottom: 0.5rem">
            <el-tag size="small" type="info">Observation</el-tag>
            <p style="white-space: pre-wrap">{{ cleanAnswer(store.selectedLog.answer.observation) }}</p>
          </div>
          <div v-if="store.selectedLog.answer.cause" style="margin-bottom: 0.5rem">
            <el-tag size="small" type="warning">Cause</el-tag>
            <p style="white-space: pre-wrap">{{ cleanAnswer(store.selectedLog.answer.cause) }}</p>
          </div>
          <div v-if="store.selectedLog.answer.impact" style="margin-bottom: 0.5rem">
            <el-tag size="small" type="danger">Impact</el-tag>
            <p style="white-space: pre-wrap">{{ cleanAnswer(store.selectedLog.answer.impact) }}</p>
          </div>
        </div>

        <p v-if="store.selectedLog.answer?.summary" style="margin-bottom: 0.75rem">
          <strong>摘要：</strong>{{ store.selectedLog.answer.summary }}
        </p>

        <el-collapse>
          <el-collapse-item title="原始 JSON">
            <pre style="white-space: pre-wrap; background: #f8f9fa; padding: 0.75rem; border-radius: 4px">{{ JSON.stringify(store.selectedLog.answer, null, 2) }}</pre>
          </el-collapse-item>
        </el-collapse>

        <div v-if="store.selectedLog.warnings.length" style="margin-top: 1rem">
          <h4>警告</h4>
          <el-alert
            v-for="w in store.selectedLog.warnings"
            :key="w"
            :title="w"
            type="warning"
            :closable="false"
            style="margin-bottom: 0.5rem"
          />
        </div>

        <div v-if="store.selectedLog.evidence.length" style="margin-top: 1rem">
          <h4>证据</h4>
          <el-collapse>
            <el-collapse-item
              v-for="(ev, idx) in store.selectedLog.evidence"
              :key="idx"
              :title="`证据 ${idx + 1}`"
            >
              <pre style="white-space: pre-wrap; font-size: 0.85rem">{{ JSON.stringify(ev, null, 2) }}</pre>
            </el-collapse-item>
          </el-collapse>
        </div>
      </div>
    </el-drawer>
  </div>
</template>
