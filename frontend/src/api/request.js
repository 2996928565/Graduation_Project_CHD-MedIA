/**
 * Axios 请求工具
 * 统一设置 BaseURL、Token 请求头，统一处理响应错误提示
 */
import axios from 'axios'
import { ElMessage } from 'element-plus'

const request = axios.create({
  baseURL: '/api/v1',
  timeout: 120000, // 120 秒（影像检测可能耗时较长）
})

// 请求拦截器：注入 Token
request.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('chd_token')
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error),
)

// 响应拦截器：统一错误提示
request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const status = error.response?.status
    const detail = error.response?.data?.detail

    const messages = {
      401: '认证失败，请重新登录',
      403: detail || '权限不足',
      404: '请求的资源不存在',
      413: '文件过大，请压缩后重试',
      422: `参数校验失败：${detail || '请检查输入'}`,
      500: `服务器错误：${detail || '请稍后重试'}`,
    }

    const msg = messages[status] || detail || `网络错误 (${status || '未知'})`
    ElMessage.error(msg)

    // 401 时跳转登录
    if (status === 401) {
      localStorage.removeItem('chd_token')
      window.location.href = '/login'
    }

    return Promise.reject(error)
  },
)

export default request
