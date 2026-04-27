<template>
  <el-card class="page-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <span>创建用户</span>
        <el-tag v-if="isAdmin" type="warning" effect="dark" size="small">仅管理员可用</el-tag>
      </div>
    </template>

    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-position="top"
      style="max-width: 520px"
      @submit.prevent="handleSubmit"
    >
      <el-form-item label="用户名" prop="username">
        <el-input v-model="form.username" placeholder="例如 doctor1" />
      </el-form-item>

      <el-form-item label="密码" prop="password">
        <el-input v-model="form.password" type="password" show-password placeholder="至少 6 位" />
      </el-form-item>

      <el-form-item label="姓名" prop="full_name">
        <el-input v-model="form.full_name" placeholder="可选" />
      </el-form-item>

      <el-form-item label="角色" prop="role">
        <el-select v-model="form.role" style="width: 100%">
          <el-option label="医生（doctor）" value="doctor" />
          <el-option label="管理员（admin）" value="admin" />
        </el-select>
      </el-form-item>

      <el-form-item>
        <el-checkbox v-model="form.is_active">启用该用户</el-checkbox>
      </el-form-item>

      <el-form-item>
        <el-button type="primary" :loading="submitting" @click="handleSubmit">创建</el-button>
        <el-button :disabled="submitting" @click="handleReset">重置</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/store/auth.js'
import { registerUser } from '@/api/auth.js'

const authStore = useAuthStore()
const isAdmin = computed(() => (authStore.role || '').toLowerCase() === 'admin')

const formRef = ref(null)
const submitting = ref(false)

const form = reactive({
  username: '',
  password: '',
  full_name: '',
  role: 'doctor',
  is_active: true,
})

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码至少 6 位', trigger: 'blur' },
  ],
  role: [{ required: true, message: '请选择角色', trigger: 'change' }],
}

function handleReset() {
  form.username = ''
  form.password = ''
  form.full_name = ''
  form.role = 'doctor'
  form.is_active = true
  formRef.value?.clearValidate?.()
}

async function handleSubmit() {
  await formRef.value?.validate(async (valid) => {
    if (!valid) return
    submitting.value = true
    try {
      const payload = {
        username: form.username.trim(),
        password: form.password,
        full_name: form.full_name?.trim() || null,
        role: form.role,
        is_active: !!form.is_active,
      }
      const res = await registerUser(payload)
      ElMessage.success(`用户创建成功：${res.full_name || res.username}`)
      form.password = ''
    } catch {
      // 错误提示由 request 拦截器统一处理
    } finally {
      submitting.value = false
    }
  })
}
</script>

<style scoped>
.page-card {
  border-radius: 12px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
</style>
