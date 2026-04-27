import request from './request.js'

/**
 * 上传并预览影像（DICOM 解析）
 * @param {File} file
 */
export function uploadPreview(file) {
  const formData = new FormData()
  formData.append('file', file)
  return request.post('/images/upload-preview', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

/**
 * 执行影像异常检测
 * @param {File} file - 影像文件
 * @param {string} modality - 'ultrasound' | 'mri'
 * @param {number} confidenceThreshold - 置信度阈值
 * @param {string|null} patientId - 患者 ID（可选）
 */
export function detectImage(file, modality, confidenceThreshold = 0.5, patientId = null) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('modality', modality)
  formData.append('confidence_threshold', String(confidenceThreshold))
  if (patientId) {
    formData.append('patient_id', patientId)
  }
  return request.post('/images/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 分钟（大影像检测）
  })
}

/**
 * 获取 NIfTI 3D 指定切片的展示结果（标注图 + mask + 检测列表）
 * @param {string} taskId
 * @param {number} sliceIndex
 * @param {number} confidenceThreshold
 */
export function getNiftiSlice(taskId, sliceIndex, confidenceThreshold = 0.5) {
  return request.get(`/images/nifti-slice/${taskId}`, {
    params: {
      slice_index: sliceIndex,
      confidence_threshold: String(confidenceThreshold),
    },
    timeout: 300000,
  })
}

/**
 * 获取检测历史记录
 * @param {Object} params
 * @param {number} params.page
 * @param {number} params.page_size
 * @param {string} params.patient_name
 * @param {string} params.doctor_name
 * @param {string} params.modality
 */
export function getDetectionHistory(params = {}) {
  return request.get('/images/history', { params })
}
