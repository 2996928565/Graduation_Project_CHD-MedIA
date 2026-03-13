import request from './request.js'

/** 用户名密码登录 */
export function login(username, password) {
  return request.post('/auth/login', { username, password })
}
