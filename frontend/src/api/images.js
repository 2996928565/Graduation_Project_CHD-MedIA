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

/**
 * 获取单条检测历史详情
 * @param {string} taskId
 */
export function getDetectionHistoryDetail(taskId) {
  return request.get(`/images/history/${taskId}`)
}

/**
 * 上传分割预测标签 zip 并启动 MRI 常模训练（MLP）
 * @param {File} zipFile
 * @param {{model_name?:string, epochs?:number, hidden_dims?:string, latent_dim?:number, batch_size?:number, lr?:number, weight_decay?:number, threshold_quantile?:number, device?:'cuda'|'cpu', pred_is_raw_mmwhs?:boolean, normal_list_name?:string}} [opts]
 */
export function startMriNormalityTrainMlp(zipFile, opts = {}) {
  const formData = new FormData()
  formData.append('dataset_zip', zipFile)
  formData.append('model_name', String(opts.model_name ?? 'mri_normal_heart_mlp'))
  formData.append('epochs', String(opts.epochs ?? 200))
  formData.append('hidden_dims', String(opts.hidden_dims ?? '64,32'))
  formData.append('latent_dim', String(opts.latent_dim ?? 8))
  formData.append('batch_size', String(opts.batch_size ?? 8))
  formData.append('lr', String(opts.lr ?? 1e-3))
  formData.append('weight_decay', String(opts.weight_decay ?? 1e-5))
  formData.append('threshold_quantile', String(opts.threshold_quantile ?? 0.99))
  formData.append('device', String(opts.device ?? 'cuda'))
  formData.append('pred_is_raw_mmwhs', String(Boolean(opts.pred_is_raw_mmwhs ?? false)))
  formData.append('normal_list_name', String(opts.normal_list_name ?? 'normal_list.txt'))
  return request.post('/normality/train-mlp', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  })
}

/** 获取 MRI 常模训练状态（MLP） */
export function getMriNormalityTrainMlpStatus(runId) {
  return request.get(`/normality/train-mlp/${runId}`)
}

/** 获取 MRI 常模训练日志（MLP，末尾若干行） */
export function getMriNormalityTrainMlpLog(runId, lines = 200) {
  return request.get(`/normality/train-mlp/${runId}/log`, { params: { lines } })
}

/** 列出已训练的常模模型（管理员） */
export function listNormalityModels(modality = 'mri') {
  return request.get('/normality/models', { params: { modality } })
}

/** 启用某个常模模型（管理员） */
export function activateNormalityModel(modelId) {
  return request.post(`/normality/models/${modelId}/activate`)
}
