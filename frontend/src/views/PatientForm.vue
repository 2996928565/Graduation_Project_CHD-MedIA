<template>
  <div>
    <div class="page-header">
      <h2>{{ isEdit ? '编辑患者信息' : '新增患者信息' }}</h2>
      <el-button @click="$router.push('/patients')">返回列表</el-button>
    </div>

    <el-card shadow="never" v-loading="fetchLoading">
      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-width="140px"
        label-position="right"
      >
        <el-divider content-position="left">基本信息</el-divider>

        <el-row :gutter="24">
          <el-col :span="12">
            <el-form-item label="患者姓名" prop="name">
              <el-input v-model="form.name" placeholder="请输入姓名" />
            </el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="年龄" prop="age">
              <el-input-number v-model="form.age" :min="0" :max="150" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="6">
            <el-form-item label="性别" prop="sex">
              <el-radio-group v-model="form.sex">
                <el-radio value="男">男</el-radio>
                <el-radio value="女">女</el-radio>
                <el-radio value="未知">未知</el-radio>
              </el-radio-group>
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="24">
          <el-col :span="12">
            <el-form-item label="联系电话">
              <el-input v-model="form.contact" placeholder="选填" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="身份证号">
              <el-input v-model="form.id_number" placeholder="选填" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="24">
          <el-col :span="12">
            <el-form-item label="申请医生">
              <el-input v-model="form.referring_doctor" placeholder="选填" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="24">
          <el-col :span="12">
            <el-form-item label="申请科室">
              <span class="fixed-value">医学影像科</span>
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="检查模态" prop="exam_modality">
              <el-select v-model="form.exam_modality" style="width:100%">
                <el-option label="心脏超声" value="ultrasound" />
                <el-option label="心脏 MRI（CMR）" value="mri" />
                <el-option label="超声 + MRI" value="both" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">先心病专属信息</el-divider>

        <el-form-item label="先心病高危因素">
          <el-checkbox-group v-model="form.chd_risk_factors">
            <el-checkbox value="母亲孕期感染风疹">母亲孕期感染风疹</el-checkbox>
            <el-checkbox value="母亲孕期糖尿病">母亲孕期糖尿病</el-checkbox>
            <el-checkbox value="家族性先心病史">家族性先心病史</el-checkbox>
            <el-checkbox value="染色体异常">染色体异常（如唐氏综合征）</el-checkbox>
            <el-checkbox value="孕期服药史">孕期服药史</el-checkbox>
            <el-checkbox value="早产">早产（孕周 &lt; 37 周）</el-checkbox>
          </el-checkbox-group>
        </el-form-item>

        <el-form-item label="主诉">
          <el-input
            v-model="form.chief_complaint"
            type="textarea"
            :rows="2"
            placeholder="如：发现心脏杂音 2 年，活动后气促 1 个月"
          />
        </el-form-item>

        <el-form-item label="既往史">
          <el-input
            v-model="form.medical_history"
            type="textarea"
            :rows="2"
            placeholder="选填"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" :loading="loading" @click="handleSubmit">
            保存患者信息
          </el-button>
          <el-button @click="$router.push('/patients')">取消</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { createPatient, getPatient, updatePatient } from '@/api/patients.js'
import { useAuthStore } from '@/store/auth.js'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const formRef = ref(null)
const loading = ref(false)
const fetchLoading = ref(false)

const isEdit = computed(() => !!route.params.id)

const form = reactive({
  name: '',
  age: null,
  sex: '未知',
  contact: '',
  id_number: '',
  referring_doctor: '',
  department: '医学影像科',
  exam_modality: 'mri',
  chd_risk_factors: [],
  chief_complaint: '',
  medical_history: '',
})

const rules = {
  name: [{ required: true, message: '请输入患者姓名', trigger: 'blur' }],
  age: [{ required: true, message: '请输入年龄', trigger: 'blur' }],
  sex: [{ required: true, message: '请选择性别', trigger: 'change' }],
  exam_modality: [{ required: true, message: '请选择检查模态', trigger: 'change' }],
}

onMounted(async () => {
  if (isEdit.value) {
    fetchLoading.value = true
    try {
      const data = await getPatient(route.params.id)
      Object.assign(form, {
        name: data.name,
        age: data.age,
        sex: data.sex,
        contact: data.contact || '',
        id_number: data.id_number || '',
        referring_doctor: data.referring_doctor || '',
        department: '医学影像科',
        exam_modality: data.exam_modality,
        chd_risk_factors: data.chd_risk_factors || [],
        chief_complaint: data.chief_complaint || '',
        medical_history: data.medical_history || '',
      })
    } catch {
      ElMessage.error('加载患者信息失败')
      router.push('/patients')
    } finally {
      fetchLoading.value = false
    }
  } else {
    if (!form.referring_doctor) {
      form.referring_doctor = authStore.fullName || authStore.username || ''
    }
  }
})

async function handleSubmit() {
  await formRef.value?.validate(async (valid) => {
    if (!valid) return
    loading.value = true
    try {
      if (isEdit.value) {
        await updatePatient(route.params.id, form)
        ElMessage.success('患者信息更新成功')
      } else {
        await createPatient(form)
        ElMessage.success('患者信息保存成功')
      }
      router.push('/patients')
    } finally {
      loading.value = false
    }
  })
}
</script>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.page-header h2 {
  margin: 0;
  color: #1a3a5c;
}
.fixed-value {
  color: #303133;
  font-size: 14px;
  line-height: 32px;
}
</style>
