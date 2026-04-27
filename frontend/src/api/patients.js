import request from './request.js'

/** 新增患者 */
export function createPatient(data) {
  return request.post('/patients', data)
}

/** 获取患者列表 */
export function getPatients(params = {}) {
  return request.get('/patients', { params })
}

/** 获取患者详情 */
export function getPatient(patientId) {
  return request.get(`/patients/${patientId}`)
}

/** 更新患者信息 */
export function updatePatient(patientId, data) {
  return request.patch(`/patients/${patientId}`, data)
}

/** 删除患者 */
export function deletePatient(patientId) {
  return request.delete(`/patients/${patientId}`)
}
