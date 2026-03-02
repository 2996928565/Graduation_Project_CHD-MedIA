import request from './request.js'
import axios from 'axios'

/**
 * 生成诊断报告
 * @param {object} payload - { modality, patient_info, detections }
 */
export function generateReport(payload) {
  return request.post('/reports/generate', payload)
}

/**
 * 导出报告为文本格式
 * @param {object} payload - { modality, patient_info, detections }
 */
export function exportReportText(payload) {
  return request.post('/reports/export/text', payload)
}

/**
 * 导出报告为 Word 文档（触发浏览器下载）
 * @param {object} payload - { modality, patient_info, detections }
 * @param {string} filename - 下载文件名
 */
export async function exportReportDocx(payload, filename = 'CHD_Report.docx') {
  const token = localStorage.getItem('chd_token')
  const response = await axios.post('/api/v1/reports/export/docx', payload, {
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    responseType: 'blob',
  })
  const url = URL.createObjectURL(response.data)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
