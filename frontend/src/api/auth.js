import request from './request.js'

/** 用户名密码登录 */
export function login(username, password) {
  return request.post('/auth/login', { username, password })
}

/** 管理员创建用户（注册） */
export function registerUser(payload) {
  return request.post('/auth/register', payload)
}

/** 用户自助注册（需要后端配置允许） */
export function registerPublic(username, password, full_name) {
  return request.post('/auth/register-public', { username, password, full_name })
}
