<template>
  <div>
    <div class="page-header">
      <h2>患者管理</h2>
      <el-button type="primary" icon="Plus" @click="$router.push('/patients/new')">
        新增患者
      </el-button>
    </div>

    <el-card shadow="never">
      <el-table
        v-loading="loading"
        :data="patients"
        stripe
        style="width: 100%"
        empty-text="暂无患者数据，请先录入患者信息"
      >
        <el-table-column prop="name" label="姓名" width="100" />
        <el-table-column prop="age" label="年龄" width="70">
          <template #default="{ row }">{{ row.age }}岁</template>
        </el-table-column>
        <el-table-column prop="sex" label="性别" width="70" />
        <el-table-column prop="exam_modality" label="检查模态" width="100">
          <template #default="{ row }">
            <el-tag :type="row.exam_modality === 'mri' ? 'warning' : 'primary'" size="small">
              {{ modalityLabel(row.exam_modality) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="department" label="科室" />
        <el-table-column prop="referring_doctor" label="申请医生" />
        <el-table-column prop="created_at" label="录入时间" width="180">
          <template #default="{ row }">{{ formatDate(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="330">
          <template #default="{ row }">
            <el-button
              v-if="false"
              size="small"
              icon="VideoCamera"
              @click="goDetect(row, 'ultrasound')"
            >超声检测</el-button>
            <el-button
              size="small"
              type="warning"
              icon="PictureFilled"
              @click="goDetect(row, 'mri')"
            >影像检测</el-button>
            <el-button
              size="small"
              type="primary"
              plain
              icon="Edit"
              @click="goEdit(row)"
            >编辑</el-button>
            <el-button
              size="small"
              type="danger"
              plain
              icon="Delete"
              @click="handleDelete(row)"
            >删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getPatients, deletePatient } from '@/api/patients.js'

const router = useRouter()
const loading = ref(false)
const patients = ref([])

onMounted(fetchPatients)

async function fetchPatients() {
  loading.value = true
  try {
    patients.value = await getPatients()
  } finally {
    loading.value = false
  }
}

function modalityLabel(m) {
  return { ultrasound: '超声', mri: '影像', both: '超声+影像' }[m] || m
}

function formatDate(iso) {
  return iso ? iso.replace('T', ' ').slice(0, 16) : ''
}

function goDetect(patient, modality) {
  const route = modality === 'mri' ? '/mri' : '/ultrasound'
  router.push({ path: route, query: { patientId: patient.patient_id, name: patient.name } })
}

function goEdit(patient) {
  router.push(`/patients/${patient.patient_id}/edit`)
}

async function handleDelete(patient) {
  try {
    await ElMessageBox.confirm(
      `确定要删除患者「${patient.name}」的信息吗？此操作不可恢复。`,
      '删除确认',
      { confirmButtonText: '确定删除', cancelButtonText: '取消', type: 'warning' }
    )
  } catch {
    return
  }
  try {
    await deletePatient(patient.patient_id)
    ElMessage.success('患者信息已删除')
    patients.value = patients.value.filter(p => p.patient_id !== patient.patient_id)
  } catch {
    ElMessage.error('删除失败，请重试')
  }
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
</style>
