<template>
  <div class="login-container">
    <div class="login-card">
      <div class="login-header">
        <img src="/favicon.svg" alt="logo" class="login-logo" />
        <h1 class="login-title">CHD-MedIA</h1>
        <p class="login-subtitle">先天性心脏病影像检测与报告生成系统</p>
      </div>

      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-position="top"
        @submit.prevent="handleLogin"
      >
        <el-form-item label="访问 Token" prop="token">
          <el-input
            v-model="form.token"
            type="password"
            placeholder="请输入访问 Token"
            show-password
            size="large"
            prefix-icon="Lock"
            @keyup.enter="handleLogin"
          />
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            size="large"
            :loading="loading"
            style="width: 100%"
            @click="handleLogin"
          >
            <el-icon><Key /></el-icon>
            登 录
          </el-button>
        </el-form-item>
      </el-form>

      <el-alert
        type="info"
        :closable="false"
        show-icon
        style="margin-top: 8px"
      >
        <template #title>
          演示 Token：<code>CHD_MEDIA_SECRET_TOKEN</code>
        </template>
      </el-alert>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/store/auth.js'
import { login } from '@/api/auth.js'

const router = useRouter()
const authStore = useAuthStore()

const formRef = ref(null)
const loading = ref(false)
const form = reactive({ token: '' })

const rules = {
  token: [{ required: true, message: '请输入访问 Token', trigger: 'blur' }],
}

async function handleLogin() {
  await formRef.value?.validate(async (valid) => {
    if (!valid) return
    loading.value = true
    try {
      const res = await login(form.token)
      authStore.setToken(res.access_token)
      ElMessage.success('登录成功，欢迎使用 CHD-MedIA 系统')
      router.push('/')
    } catch {
      // 错误已在拦截器统一处理
    } finally {
      loading.value = false
    }
  })
}
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #e8f4fd 0%, #d1eaff 50%, #c3e0ff 100%);
}

.login-card {
  width: 420px;
  padding: 48px 40px;
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 8px 40px rgba(0, 100, 200, 0.12);
}

.login-header {
  text-align: center;
  margin-bottom: 32px;
}

.login-logo {
  width: 64px;
  height: 64px;
  margin-bottom: 12px;
}

.login-title {
  font-size: 28px;
  font-weight: 700;
  color: #1a3a5c;
  margin: 0 0 6px;
}

.login-subtitle {
  font-size: 14px;
  color: #5a7fa0;
  margin: 0;
}
</style>
