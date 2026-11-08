import api from './client'
import type { DocumentOut } from './types'

export async function listDocuments(): Promise<DocumentOut[]> {
  const res = await api.get('/documents')
  return res.data
}

export async function uploadDocument(
  file: File,
  year = '',
  department = '',
): Promise<any> {
  const form = new FormData()
  form.append('file', file)
  form.append('year', year)
  form.append('department', department)
  const res = await api.post('/documents/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export async function deleteDocument(docId: string): Promise<any> {
  const res = await api.delete(`/documents/${docId}`)
  return res.data
}

export async function parseDocument(docId: string): Promise<any> {
  const res = await api.post(`/documents/${docId}/parse`)
  return res.data
}

export async function parseAllDocuments(): Promise<any> {
  const res = await api.post('/documents/parse-all')
  return res.data
}
