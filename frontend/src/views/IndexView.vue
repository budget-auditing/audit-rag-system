<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, computed } from 'vue'
import { ElMessage } from 'element-plus'
import {
  getIndexStatus,
  getBuildStatus,
  buildAllIndexes,
  buildFactGraph,
  buildInstitutionalIndex,
  buildRiskIndex,
} from '@/api/index'
import type { IndexStatus, BuildJob, BuildIndexResponse } from '@/api/types'

const status = ref<IndexStatus | null>(null)
const job = ref<BuildJob | null>(null)
let timer: ReturnType<typeof setInterval> | null = null

const building = computed(
  () => job.value?.status === 'queued' || job.value?.status === 'building',
)

const percent = computed(() => {
  const p = job.value?.progress
  if (!p || !p.total) return job.value?.status === 'ready' ? 100 : 0
  return Math.min(100, Math.round((p.current / p.total) * 100))
})

const progressStatus = computed(() => {
  if (job.value?.status === 'failed') return 'exception'
  if (job.value?.status === 'ready') return 'success'
  return undefined
})

function formatTime(iso?: string): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (isNaN(d.getTime())) return iso
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

async function load() {
  status.value = await getIndexStatus()
}

function stopPolling() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

function startPolling(jobId: string) {
  stopPolling()
  timer = setInterval(async () => {
    const res = await getBuildStatus(jobId)
    job.value = res.job
    if (!res.active) {
      stopPolling()
      await load()
      if (res.job?.status === 'ready') ElMessage.success('索引构建完成')
      else if (res.job?.status === 'failed') ElMessage.error(res.job?.error || '索引构建失败')
    }
  }, 2000)
}

async function kick(fn: () => Promise<BuildIndexResponse>) {
  try {
    const res = await fn()
    if (res.job) {
      job.value = res.job
      startPolling(res.job.job_id)
    } else {
      ElMessage.warning(res.message || '无可构建内容')
    }
  } catch (e: any) {
    const detail = e?.response?.data?.detail
    if (detail?.job) {
      job.value = detail.job
      startPolling(detail.job.job_id)
      ElMessage.warning(detail.message || '已有任务进行中')
    } else {
      ElMessage.error(detail?.message || detail || '启动失败')
    }
  }
}

const buildAll = () => kick(buildAllIndexes)
const buildFact = () => kick(buildFactGraph)
const buildInstitutional = () => kick(buildInstitutionalIndex)
const buildRisk = () => kick(buildRiskIndex)

onMounted(async () => {
  await load()
  // Resume display if a build is already running (e.g. after page reload).
  const res = await getBuildStatus()
  job.value = res.job
  if (res.active && res.job) startPolling(res.job.job_id)
})

onBeforeUnmount(stopPolling)
</script>

<template>
  <div>
    <h2>索引构建</h2>
    <el-space wrap>
      <el-button type="primary" :loading="building" @click="buildAll">一键构建全部索引</el-button>
      <el-button :disabled="building" @click="buildFact">构建多层语义图</el-button>
      <el-button :disabled="building" @click="buildInstitutional">构建结论核验证据索引</el-button>
      <el-button :disabled="building" @click="buildRisk">构建风险索引</el-button>
      <el-button @click="load">刷新状态</el-button>
    </el-space>

    <div v-if="job" style="margin-top: 1rem">
      <div style="margin-bottom: 0.5rem">
        <el-tag
          :type="job.status === 'ready' ? 'success' : job.status === 'failed' ? 'danger' : 'warning'"
          size="small"
        >
          {{ job.status }}
        </el-tag>
        <span style="margin-left: 0.5rem; color: #606266">
          目标: {{ job.target }}
          <template v-if="job.phase"> · {{ job.phase }}</template>
          <template v-if="job.progress?.total">
            （{{ job.progress.current }}/{{ job.progress.total }}）
          </template>
        </span>
      </div>
      <el-progress :percentage="percent" :status="progressStatus" :stroke-width="14" />
      <p v-if="job.error" style="color: #f56c6c; margin-top: 0.5rem">{{ job.error }}</p>
    </div>

    <div v-if="status" style="margin-top: 1.5rem">
      <el-card v-if="status.corpus" shadow="never" style="margin-bottom: 1rem">
        <template #header>审计材料解析结果</template>
        <el-descriptions :column="3" size="small" border>
          <el-descriptions-item label="标准化片段">{{ status.corpus.chunks_total ?? '-' }}</el-descriptions-item>
          <el-descriptions-item label="表格数量">{{ status.corpus.tables_total ?? '-' }}</el-descriptions-item>
          <el-descriptions-item label="解析耗时(s)">{{ status.corpus.parse_seconds ?? '-' }}</el-descriptions-item>
        </el-descriptions>
      </el-card>
      <el-row :gutter="16">
        <el-col :span="8">
          <el-card shadow="never">
            <template #header>多层语义图索引 (MSG-RAG)</template>
            <el-descriptions :column="1" size="small" border>
              <el-descriptions-item label="状态">
                <el-tag :type="status.msg?.built ? 'success' : 'info'" size="small">
                  {{ status.msg?.built ? '已构建' : '未构建' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="结构节点">{{ status.msg?.structure_count ?? 0 }}</el-descriptions-item>
              <el-descriptions-item label="实例节点">{{ status.msg?.count ?? 0 }}</el-descriptions-item>
              <el-descriptions-item label="边数">{{ status.msg?.edges ?? 0 }}</el-descriptions-item>
              <el-descriptions-item v-if="status.msg?.build_seconds" label="构建耗时(s)">
                {{ status.msg.build_seconds }}
              </el-descriptions-item>
            </el-descriptions>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card shadow="never">
            <template #header>结论核验证据索引 (C³-RAG)</template>
            <el-descriptions :column="1" size="small" border>
              <el-descriptions-item label="状态">
                <el-tag :type="status.c3rag?.ready ? 'success' : 'info'" size="small">
                  {{ status.c3rag?.ready ? '就绪' : '未就绪' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="证据条数">
                {{ status.c3rag?.count ?? status.c3rag?.size ?? status.c3rag?.num_docs ?? '-' }}
              </el-descriptions-item>
              <el-descriptions-item v-if="status.c3rag?.field_stats" label="字段覆盖">
                年度 {{ status.c3rag.field_stats.years }} · 部门 {{ status.c3rag.field_stats.departments }} · 指标 {{ status.c3rag.field_stats.metrics }}
              </el-descriptions-item>
              <el-descriptions-item v-if="status.c3rag?.build_seconds" label="构建耗时(s)">
                {{ status.c3rag.build_seconds }}
              </el-descriptions-item>
              <el-descriptions-item v-if="status.c3rag?.error" label="错误">
                {{ status.c3rag.error }}
              </el-descriptions-item>
            </el-descriptions>
          </el-card>
        </el-col>
        <el-col :span="8">
          <el-card shadow="never">
            <template #header>风险敏感片段索引 (TRACE)</template>
            <el-descriptions :column="1" size="small" border>
              <el-descriptions-item label="状态">
                <el-tag :type="status.trace?.ready ? 'success' : 'info'" size="small">
                  {{ status.trace?.ready ? '就绪' : '未就绪' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="段落条数">
                {{ status.trace?.count ?? status.trace?.size ?? status.trace?.num_docs ?? '-' }}
              </el-descriptions-item>
              <el-descriptions-item v-if="status.trace?.build_seconds" label="构建耗时(s)">
                {{ status.trace.build_seconds }}
              </el-descriptions-item>
              <el-descriptions-item v-if="status.trace?.error" label="错误">
                {{ status.trace.error }}
              </el-descriptions-item>
            </el-descriptions>
          </el-card>
        </el-col>
      </el-row>
      <p style="color: #909399; font-size: 0.8rem; margin-top: 0.5rem">
        更新时间：{{ formatTime(status.last_update) }}
      </p>
      <el-collapse style="margin-top: 0.5rem">
        <el-collapse-item title="原始状态 JSON">
          <pre style="background: #f8f9fa; padding: 1rem; border-radius: 4px">{{ JSON.stringify(status, null, 2) }}</pre>
        </el-collapse-item>
      </el-collapse>
    </div>
  </div>
</template>
