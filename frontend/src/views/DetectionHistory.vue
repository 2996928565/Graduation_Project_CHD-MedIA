<template>
  <div>
    <div class="page-header">
      <h2>检测历史</h2>
    </div>

    <el-card shadow="never" class="search-card">
      <el-form :inline="true" :model="filters" @submit.prevent>
        <el-form-item label="患者姓名">
          <el-input
            v-model="filters.patient_name"
            placeholder="输入患者姓名关键字"
            clearable
            @keyup.enter="fetchHistory"
          />
        </el-form-item>
        <el-form-item v-if="isAdmin" label="医生姓名">
          <el-input
            v-model="filters.doctor_name"
            placeholder="输入医生姓名关键字"
            clearable
            @keyup.enter="fetchHistory"
          />
        </el-form-item>
        <el-form-item label="检查模态">
          <el-select v-model="filters.modality" clearable placeholder="全部" style="width: 150px">
            <el-option label="超声" value="ultrasound" />
            <el-option label="影像" value="mri" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">搜索</el-button>
          <el-button @click="handleReset">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never">
      <el-table
        v-loading="loading"
        :data="rows"
        stripe
        style="width: 100%"
        empty-text="暂无检测历史"
      >
        <el-table-column prop="task_id" label="任务ID" min-width="180" show-overflow-tooltip />
        <el-table-column prop="patient_name" label="患者姓名" width="110" />
        <el-table-column prop="doctor_name" label="检测医生" width="120" />
        <el-table-column prop="modality" label="模态" width="90">
          <template #default="{ row }">
            <el-tag :type="row.modality === 'mri' ? 'warning' : 'primary'" size="small">
              {{ row.modality === 'mri' ? '影像' : '超声' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="filename" label="文件名" min-width="180" show-overflow-tooltip />
        <el-table-column prop="detections_count" label="检测项" width="80" />
        <el-table-column prop="processing_time_s" label="耗时(s)" width="90">
          <template #default="{ row }">{{ formatSeconds(row.processing_time_s) }}</template>
        </el-table-column>
        <el-table-column prop="created_at" label="检测时间" width="180">
          <template #default="{ row }">{{ formatDate(row.created_at) }}</template>
        </el-table-column>
      </el-table>

      <div class="pager-wrap">
        <el-pagination
          v-model:current-page="pager.page"
          v-model:page-size="pager.page_size"
          background
          layout="total, sizes, prev, pager, next, jumper"
          :page-sizes="[10, 20, 50]"
          :total="pager.total"
          @current-change="fetchHistory"
          @size-change="handleSizeChange"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useAuthStore } from '@/store/auth.js'
import { getDetectionHistory } from '@/api/images.js'

const authStore = useAuthStore()
const isAdmin = computed(() => (authStore.role || '').toLowerCase() === 'admin')

const loading = ref(false)
const rows = ref([])
const filters = ref({
  patient_name: '',
  doctor_name: '',
  modality: '',
})
const pager = ref({
  page: 1,
  page_size: 20,
  total: 0,
})

onMounted(fetchHistory)

async function fetchHistory() {
  loading.value = true
  try {
    const params = {
      page: pager.value.page,
      page_size: pager.value.page_size,
      patient_name: (filters.value.patient_name || '').trim(),
      modality: filters.value.modality || '',
    }
    if (isAdmin.value) {
      params.doctor_name = (filters.value.doctor_name || '').trim()
    }

    const res = await getDetectionHistory(params)
    rows.value = res.items || []
    pager.value.total = res.total || 0
  } finally {
    loading.value = false
  }
}

function handleSearch() {
  pager.value.page = 1
  fetchHistory()
}

function handleReset() {
  filters.value.patient_name = ''
  filters.value.doctor_name = ''
  filters.value.modality = ''
  pager.value.page = 1
  fetchHistory()
}

function handleSizeChange() {
  pager.value.page = 1
  fetchHistory()
}

function formatDate(iso) {
  return iso ? iso.replace('T', ' ').slice(0, 19) : ''
}

function formatSeconds(v) {
  const n = Number(v || 0)
  return Number.isFinite(n) ? n.toFixed(2) : '0.00'
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

.search-card {
  margin-bottom: 16px;
}

.pager-wrap {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
}
</style>
