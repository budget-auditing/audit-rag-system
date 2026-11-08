import { defineStore } from 'pinia'
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import type { DocumentOut } from '@/api/types'
import * as documentsApi from '@/api/documents'

function errText(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
  if (detail) return detail
  return e instanceof Error ? e.message : String(e)
}

export const useDocumentsStore = defineStore('documents', () => {
  const docs = ref<DocumentOut[]>([])
  const loading = ref(false)

  async function load() {
    loading.value = true
    try {
      docs.value = await documentsApi.listDocuments()
    } finally {
      loading.value = false
    }
  }

  async function upload(file: File, year = '', department = '') {
    await documentsApi.uploadDocument(file, year, department)
    await load()
  }

  async function uploadMany(files: File[], year = '', department = '') {
    let ok = 0
    const failed: string[] = []
    for (const file of files) {
      try {
        await documentsApi.uploadDocument(file, year, department)
        ok += 1
      } catch (e) {
        failed.push(`${file.name}（${errText(e)}）`)
      }
    }
    await load()
    if (ok) ElMessage.success(`成功上传 ${ok} 个文件`)
    if (failed.length) ElMessage.error(`上传失败 ${failed.length} 个：${failed.join('，')}`)
  }

  async function remove(docId: string) {
    await documentsApi.deleteDocument(docId)
    await load()
  }

  async function parse(docId: string) {
    try {
      const res = await documentsApi.parseDocument(docId)
      ElMessage.success(`解析完成：${(res as { chunks?: number })?.chunks ?? 0} 个分块`)
    } catch (e) {
      ElMessage.error(`解析失败：${errText(e)}`)
    } finally {
      await load()
    }
  }

  async function parseAll() {
    loading.value = true
    try {
      const res = await documentsApi.parseAllDocuments()
      const { total = 0, parsed = 0, failed = 0 } = res as {
        total?: number
        parsed?: number
        failed?: number
      }
      if (total === 0) {
        ElMessage.info('没有待解析的文档')
      } else if (failed === 0) {
        ElMessage.success(`已解析 ${parsed} 个文档`)
      } else {
        ElMessage.warning(`解析完成：成功 ${parsed} 个，失败 ${failed} 个`)
      }
    } catch (e) {
      ElMessage.error(`批量解析失败：${errText(e)}`)
    } finally {
      await load()
    }
  }

  return { docs, loading, load, upload, uploadMany, remove, parse, parseAll }
})
