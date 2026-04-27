<template>
  <div class="login-container">
    <div class="login-card">
      <div class="login-header">
        <img src="/favicon.svg" alt="logo" class="login-logo" />
        <h1 class="login-title">CHD-MedIA</h1>
        <p class="login-subtitle">先天性心脏病影像检测与报告生成系统</p>
      </div>

      <el-tabs v-model="activeTab" stretch class="login-tabs">
        <el-tab-pane label="登录" name="login" />
        <el-tab-pane label="注册" name="register" />
      </el-tabs>

      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-position="top"
        @submit.prevent="handleSubmit"
      >
        <el-form-item label="用户名" prop="username">
          <el-input
            v-model="form.username"
            placeholder="请输入用户名"
            size="large"
            prefix-icon="User"
            @keyup.enter="handleSubmit"
          />
        </el-form-item>

        <el-form-item label="密码" prop="password">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="请输入密码"
            show-password
            size="large"
            prefix-icon="Lock"
            @keyup.enter="handleSubmit"
          />
        </el-form-item>

        <el-form-item v-if="isRegister" label="确认密码" prop="confirmPassword">
          <el-input
            v-model="form.confirmPassword"
            type="password"
            placeholder="请再次输入密码"
            show-password
            size="large"
            prefix-icon="Lock"
            @keyup.enter="handleSubmit"
          />
        </el-form-item>

        <el-form-item v-if="isRegister" label="姓名（可选）" prop="fullName">
          <el-input
            v-model="form.fullName"
            placeholder="昵称"
            size="large"
            prefix-icon="User"
            @keyup.enter="handleSubmit"
          />
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            size="large"
            :loading="loading"
            style="width: 100%"
            @click="handleSubmit"
          >
            {{ isRegister ? '注 册' : '登 录' }}
          </el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/store/auth.js'
import { login, registerPublic } from '@/api/auth.js'

const router = useRouter()
const authStore = useAuthStore()

const formRef = ref(null)
const loading = ref(false)
const activeTab = ref('login')
const isRegister = computed(() => activeTab.value === 'register')

const form = reactive({
  username: '',
  password: '',
  confirmPassword: '',
  fullName: '',
})

watch(activeTab, () => {
  form.password = ''
  form.confirmPassword = ''
  formRef.value?.clearValidate?.()
})

const rules = computed(() => {
  const base = {
    username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
    password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
  }

  if (!isRegister.value) {
    return base
  }

  return {
    ...base,
    password: [
      { required: true, message: '请输入密码', trigger: 'blur' },
      { min: 6, message: '密码至少 6 位', trigger: 'blur' },
    ],
    confirmPassword: [
      { required: true, message: '请再次输入密码', trigger: 'blur' },
      {
        validator: (_rule, value, callback) => {
          if (!value) return callback(new Error('请再次输入密码'))
          if (value !== form.password) return callback(new Error('两次密码不一致'))
          return callback()
        },
        trigger: 'blur',
      },
    ],
  }
})

async function handleSubmit() {
  await formRef.value?.validate(async (valid) => {
    if (!valid) return
    loading.value = true
    try {
      if (isRegister.value) {
        await registerPublic(
          form.username.trim(),
          form.password,
          form.fullName?.trim() || null,
        )
        ElMessage.success('注册成功，正在为你登录...')
      }

      const res = await login(form.username, form.password)
      authStore.setAuth(res)
      ElMessage.success(`欢迎回来，${res.full_name || res.username}`)
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

.login-tabs {
  margin-bottom: 18px;
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
