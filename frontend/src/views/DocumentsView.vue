<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { CopyDocument } from '@element-plus/icons-vue'
import { useDocumentsStore } from '@/store/documents'

const store = useDocumentsStore()
const fileInput = ref<HTMLInputElement | null>(null)
const uploadYear = ref('')
const uploadDept = ref('')

onMounted(() => store.load())

const hasUnparsed = computed(() =>
  store.docs.some((d) => d.parse_status !== 'parsed'),
)

function onFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files && target.files.length) {
    store.uploadMany(Array.from(target.files), uploadYear.value, uploadDept.value)
    target.value = ''
  }
}

function parseStatus(raw: string | null | undefined): Record<string, string> {
  if (!raw) return {}
  try {
    return JSON.parse(raw) as Record<string, string>
  } catch {
    return { raw: raw }
  }
}

function indexKeys(raw: string | null | undefined): string[] {
  return Object.keys(parseStatus(raw)).filter((k) => k !== 'last_update')
}

function statusValue(raw: string | null | undefined, key: string): string {
  return parseStatus(raw)[key] ?? ''
}

function badgeType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  switch (status) {
    case 'built':
      return 'success'
    case 'building':
      return 'warning'
    case 'failed':
      return 'danger'
    default:
      return 'info'
  }
}

function shortId(id: string): string {
  return id.slice(0, 8)
}

async function copyId(id: string) {
  try {
    await navigator.clipboard.writeText(id)
    ElMessage.success('ID 已复制')
  } catch {
    ElMessage.error('复制失败')
  }
}
</script>

<template>
  <div>
    <h2>文档管理</h2>
    <input ref="fileInput" type="file" multiple style="display:none" @change="onFileChange" />
    <el-space style="margin-bottom: 0.5rem" wrap>
      <el-input v-model="uploadYear" placeholder="年度（可选，如 2024）" style="width: 180px" clearable />
      <el-input v-model="uploadDept" placeholder="部门（可选）" style="width: 200px" clearable />
    </el-space>
    <br />
    <el-button type="primary" @click="fileInput?.click()">上传文档</el-button>
    <el-button type="success" :disabled="!hasUnparsed" :loading="store.loading" @click="store.parseAll()">
      解析全部
    </el-button>
    <el-button @click="store.load()">刷新</el-button>

    <el-table :data="store.docs" v-loading="store.loading" style="margin-top: 1rem" border>
      <el-table-column label="ID" width="120">
        <template #default="{ row }">
          <div style="display:flex;align-items:center;gap:4px">
            <el-tooltip :content="row.doc_id" placement="top">
              <code style="font-size:12px;cursor:default">{{ shortId(row.doc_id) }}</code>
            </el-tooltip>
            <el-button size="small" link style="padding:0;height:auto" @click="copyId(row.doc_id)">
              <el-icon><CopyDocument /></el-icon>
            </el-button>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="doc_name" label="名称" />
      <el-table-column prop="doc_type" label="类型" width="90" />
      <el-table-column prop="year" label="年度" width="90" />
      <el-table-column prop="department" label="部门" width="140" show-overflow-tooltip />
      <el-table-column prop="parse_status" label="解析状态" width="100" />
      <el-table-column label="索引状态" width="200">
        <template #default="{ row }">
          <div v-if="row.index_status" style="display:flex;flex-wrap:wrap;gap:4px">
            <el-tag
              v-for="key in indexKeys(row.index_status)"
              :key="key"
              size="small"
              :type="badgeType(statusValue(row.index_status, key))"
            >
              {{ key }}:{{ statusValue(row.index_status, key) }}
            </el-tag>
          </div>
          <el-tag v-else size="small" type="info">未构建</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="180">
        <template #default="{ row }">
          <el-button size="small" @click="store.parse(row.doc_id)">解析</el-button>
          <el-button size="small" type="danger" @click="store.remove(row.doc_id)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>
