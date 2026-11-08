<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getOverview } from '@/api/stats'
import type { OverviewResponse } from '@/api/types'

const data = ref<OverviewResponse | null>(null)
const loading = ref(false)

async function load() {
  loading.value = true
  try {
    data.value = await getOverview()
  } finally {
    loading.value = false
  }
}

onMounted(load)

function jobTagType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'ready' || status === 'done') return 'success'
  if (status === 'failed') return 'danger'
  if (status === 'building' || status === 'queued') return 'warning'
  return 'info'
}

function taskTagType(type: string): 'success' | 'warning' | 'info' {
  if (type === 'index_build') return 'warning'
  if (type === 'qa') return 'success'
  return 'info'
}

function formatTime(iso: string): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}
</script>

<template>
  <div v-loading="loading">
    <h2>系统概览</h2>
    <el-row :gutter="16" v-if="data">
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-label">文档总数</div>
          <div class="stat-num">{{ data.documents.total }}</div>
          <div class="stat-sub">已解析 {{ data.documents.parsed }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-label">多层语义图索引 (MSG-RAG)</div>
          <div class="stat-num">{{ data.indexes.msg?.fact_nodes ?? 0 }}</div>
          <div class="stat-sub">
            结构节点 {{ data.indexes.msg?.structure_nodes ?? 0 }} · 边 {{ data.indexes.msg?.edges ?? 0 }}
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-label">结论核验证据索引 (C³-RAG)</div>
          <div class="stat-num">{{ data.indexes.institutional_evidence ?? 0 }}</div>
          <div class="stat-sub">口径一致性证据条数 {{ data.indexes.institutional_evidence ?? 0 }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-label">风险敏感片段索引 (TRACE)</div>
          <div class="stat-num">{{ data.indexes.risk_passages ?? 0 }}</div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card shadow="hover">
          <div class="stat-label">问答记录</div>
          <div class="stat-num">{{ data.queries.total }}</div>
          <div class="stat-sub">累计查询次数</div>
        </el-card>
      </el-col>
    </el-row>

    <h3 style="margin-top: 1.5rem">最近任务</h3>
    <el-table :data="data?.recent_tasks || []" border size="small">
      <el-table-column label="任务类型" width="120">
        <template #default="{ row }">
          <el-tag :type="taskTagType(row.task_type)" size="small">{{ row.task_type_cn }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="target" label="目标/内容" min-width="200" show-overflow-tooltip />
      <el-table-column label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="jobTagType(row.status)" size="small">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="detail" label="详情" width="180" show-overflow-tooltip />
      <el-table-column label="创建时间" width="180">
        <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
      </el-table-column>
    </el-table>
  </div>
</template>

<style scoped>
.el-row :deep(.el-card__body) { min-height: 110px; }
.stat-label { color: #909399; font-size: 0.85rem; }
.stat-num { font-size: 1.8rem; font-weight: bold; margin: 0.25rem 0; }
.stat-sub { color: #606266; font-size: 0.8rem; }
</style>
