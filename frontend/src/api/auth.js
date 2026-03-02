import request from './request.js'

/** Token 登录 */
export function login(token) {
  return request.post('/auth/login', { token })
}
