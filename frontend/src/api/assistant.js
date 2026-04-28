import request from './request.js'

/**
 * 导入某个患者的检测结果上下文
 * @param {string} patientId
 * @param {number} limit
 */
export function getPatientDetectionContext(patientId, limit = 10) {
  return request.get(`/assistant/patient-context/${patientId}`, {
    params: { limit },
  })
}

/**
 * 智能问答
 * @param {object} payload
 * @param {string} payload.question
 * @param {string} [payload.patient_id]
 * @param {string[]} [payload.task_ids]
 * @param {Array<{role:'user'|'assistant', content:string}>} [payload.history]
 */
export function chatWithAssistant(payload) {
  return request.post('/assistant/chat', payload)
}
