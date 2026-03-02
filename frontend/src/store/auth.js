import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAuthStore = defineStore('auth', () => {
  const token = ref(localStorage.getItem('chd_token') || '')

  const isLoggedIn = computed(() => !!token.value)

  function setToken(newToken) {
    token.value = newToken
    localStorage.setItem('chd_token', newToken)
  }

  function logout() {
    token.value = ''
    localStorage.removeItem('chd_token')
  }

  return { token, isLoggedIn, setToken, logout }
})
