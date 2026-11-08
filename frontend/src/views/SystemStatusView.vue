<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { getSystemStatus } from '@/api/stats'
import type { SystemStatusResponse } from '@/api/types'

const data = ref<SystemStatusResponse | null>(null)
const loading = ref(false)

async function load() {
  loading.value = true
  try {
    data.value = await getSystemStatus()
  } finally {
    loading.value = false
  }
}

onMounted(load)

const OK = new Set(['connected', 'ready', 'loaded'])

function statusType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  if (OK.has(status)) return 'success'
  if (status === 'disabled') return 'info'
  return 'danger'
}

// 组件行（排除标量展示字段）
const componentRows = computed(() => {
  const c = data.value?.components || {}
  const rows: { key: string; label: string; status: string; extra: string }[] = []
  for (const [key, val] of Object.entries(c)) {
    if (typeof val !== 'object' || val === null) continue
    const v = val as Record<string, any>
    const extras: string[] = []
    if (v.name) extras.push(v.name)
    if (v.nodes !== undefined) extras.push(`节点 ${v.nodes}`)
    if (v.edges !== undefined) extras.push(`边 ${v.edges}`)
    rows.push({ key, label: v.label || key, status: v.status || '-', extra: extras.join(' · ') })
  }
  return rows
})
</script>

<template>
  <div v-loading="loading">
    <h2>系统状态</h2>
    <el-button size="small" @click="load" style="margin-bottom: 1rem">刷新</el-button>

    <el-table :data="componentRows" border>
      <el-table-column prop="label" label="组件" width="220" />
      <el-table-column label="状态" width="140">
        <template #default="{ row }">
          <el-tag :type="statusType(row.status)">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="extra" label="详情" />
    </el-table>
  </div>
</template>
