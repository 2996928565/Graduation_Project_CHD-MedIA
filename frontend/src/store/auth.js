import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('chd_token') || '')
  const username = ref(localStorage.getItem('chd_username') || '')
  const fullName = ref(localStorage.getItem('chd_full_name') || '')
  const role = ref(localStorage.getItem('chd_role') || '')

  const isLoggedIn = computed(() => !!token.value)

  function setAuth(loginRes) {
    token.value = loginRes.access_token
    username.value = loginRes.username || ''
    fullName.value = loginRes.full_name || ''
    role.value = loginRes.role || ''
    localStorage.setItem('chd_token', loginRes.access_token)
    localStorage.setItem('chd_username', loginRes.username || '')
    localStorage.setItem('chd_full_name', loginRes.full_name || '')
    localStorage.setItem('chd_role', loginRes.role || '')
  }

  function logout() {
    token.value = ''
    username.value = ''
    fullName.value = ''
    role.value = ''
    localStorage.removeItem('chd_token')
    localStorage.removeItem('chd_username')
    localStorage.removeItem('chd_full_name')
    localStorage.removeItem('chd_role')
  }

  return { token, username, fullName, role, isLoggedIn, setAuth, logout }
})
