<template>
  <div>
    <div class="page-header">
      <h2><el-icon><VideoCamera /></el-icon> 心脏超声检测</h2>
      <div class="page-actions">
        <el-button
          v-if="hasContent"
          type="danger"
          plain
          size="small"
          @click="clearCurrentState"
        >
          清空当前内容
        </el-button>
        <el-tag type="primary">超声心动图 / 二维超声</el-tag>
      </div>
    </div>

    <!-- 患者选择 -->
    <el-card shadow="never" style="margin-bottom:16px">
      <template #header><span class="card-title">患者信息</span></template>
      <el-row :gutter="16">
        <el-col :span="8">
          <el-form-item label="患者姓名" style="margin:0">
            <el-input v-model="patientName" placeholder="输入患者姓名（用于报告）" />
          </el-form-item>
        </el-col>
        <el-col :span="4">
          <el-form-item label="年龄" style="margin:0">
            <el-input-number v-model="patientAge" :min="0" style="width:100%" />
          </el-form-item>
        </el-col>
        <el-col :span="4">
          <el-form-item label="性别" style="margin:0">
            <el-select v-model="patientSex">
              <el-option label="男" value="男" />
              <el-option label="女" value="女" />
              <el-option label="未知" value="未知" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="4">
          <el-form-item label="置信度阈值" style="margin:0">
            <el-slider v-model="threshold" :min="0.1" :max="0.9" :step="0.05" :format-tooltip="v => (v*100).toFixed(0)+'%'" />
          </el-form-item>
        </el-col>
      </el-row>
    </el-card>

    <!-- 影像上传与检测 -->
    <el-row :gutter="16">
      <el-col :span="10">
        <el-card shadow="never">
          <template #header><span class="card-title">上传超声影像</span></template>
          <ImageUpload
            ref="uploadRef"
            modality="ultrasound"
            accept=".png,.jpg,.jpeg,.dcm,.dicom"
            @file-selected="onFileSelected"
          />
          <div v-if="previewSrc" style="margin-top:12px">
            <p style="color:#5a7fa0;font-size:13px;margin:0 0 8px">原始影像预览：</p>
            <img :src="previewSrc" style="max-width:100%;border-radius:8px;border:1px solid #e0eaf5" />
          </div>
        </el-card>
      </el-col>

      <el-col :span="14">
        <el-card shadow="never">
          <template #header>
            <span class="card-title">检测结果</span>
            <el-button
              v-if="selectedFile"
              type="primary"
              size="small"
              :loading="detecting"
              style="float:right"
              @click="runDetection"
            >
              <el-icon><Search /></el-icon> 开始检测
            </el-button>
          </template>

          <div v-if="detecting" class="detecting-tip">
            <el-icon class="is-loading"><Loading /></el-icon>
            正在分析超声影像，请稍候...
          </div>

          <DetectionResult
            v-else-if="detectionResult"
            :result="detectionResult"
            modality="ultrasound"
          />

          <el-empty v-else description="请上传超声影像并点击「开始检测」" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 生成报告按钮 -->
    <div v-if="detectionResult" style="margin-top:16px;text-align:right">
      <el-button type="success" icon="Document" @click="goToReport">
        生成诊断报告
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { detectImage, uploadPreview } from '@/api/images.js'
import ImageUpload from '@/components/ImageUpload.vue'
import DetectionResult from '@/components/DetectionResult.vue'

const route = useRoute()
const router = useRouter()
const CACHE_KEY = 'chd_ultrasound_detection_state_v1'

function parseQueryAge(v) {
  if (v === undefined || v === null || v === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

function normalizeSex(v) {
  return ['男', '女', '未知'].includes(v) ? v : '未知'
}

const patientId = ref(route.query.patientId || '')
const patientName = ref(route.query.name || '')
const patientAge = ref(parseQueryAge(route.query.age))
const patientSex = ref(normalizeSex(route.query.sex))
const threshold = ref(0.5)

const uploadRef = ref(null)
const selectedFile = ref(null)
const previewSrc = ref('')
const detecting = ref(false)
const detectionResult = ref(null)
const hasRoutePrefill = computed(
  () => Boolean(patientId.value || route.query.name || route.query.age || route.query.sex),
)
const hasContent = computed(
  () => Boolean(
    selectedFile.value ||
    previewSrc.value ||
    detectionResult.value ||
    patientId.value ||
    patientName.value ||
    patientAge.value !== null ||
    (patientSex.value && patientSex.value !== '未知')
  ),
)

function onFileSelected({ file, previewBase64 }) {
  selectedFile.value = file
  previewSrc.value = !file
    ? ''
    : previewBase64
      ? `data:image/png;base64,${previewBase64}`
      : URL.createObjectURL(file)
  detectionResult.value = null
}

function clearCurrentState() {
  selectedFile.value = null
  previewSrc.value = ''
  detectionResult.value = null
  detecting.value = false
  patientId.value = ''
  patientName.value = ''
  patientAge.value = null
  patientSex.value = '未知'
  threshold.value = 0.5
  uploadRef.value?.clearFile?.()
  sessionStorage.removeItem(CACHE_KEY)
  router.replace({ path: route.path, query: {} })
  ElMessage.success('已清空当前检测内容')
}

async function runDetection() {
  if (!selectedFile.value) {
    ElMessage.warning('请先上传超声影像')
    return
  }
  detecting.value = true
  try {
    const result = await detectImage(selectedFile.value, 'ultrasound', threshold.value, patientId.value || null)
    detectionResult.value = result
    ElMessage.success(`检测完成，发现 ${result.detections.length} 条结果`)
  } finally {
    detecting.value = false
  }
}

function goToReport() {
  router.push({
    path: '/report',
    query: {
      modality: 'ultrasound',
      patientId: patientId.value || '',
      name: patientName.value,
      age: patientAge.value,
      sex: patientSex.value,
      detections: JSON.stringify(detectionResult.value?.detections || []),
    },
  })
}

function savePageState() {
  const payload = {
    patientId: patientId.value || '',
    patientName: patientName.value || '',
    patientAge: patientAge.value,
    patientSex: patientSex.value || '未知',
    threshold: threshold.value,
    previewSrc: previewSrc.value || '',
    detectionResult: detectionResult.value || null,
  }
  sessionStorage.setItem(CACHE_KEY, JSON.stringify(payload))
}

function restorePageState() {
  const raw = sessionStorage.getItem(CACHE_KEY)
  if (!raw) return
  try {
    const state = JSON.parse(raw)
    patientId.value = state.patientId || ''
    patientName.value = state.patientName || ''
    patientAge.value = parseQueryAge(state.patientAge)
    patientSex.value = normalizeSex(state.patientSex)
    threshold.value = Number.isFinite(Number(state.threshold)) ? Number(state.threshold) : 0.5
    previewSrc.value = state.previewSrc || ''
    detectionResult.value = state.detectionResult || null
  } catch {
    sessionStorage.removeItem(CACHE_KEY)
  }
}

onMounted(() => {
  if (!hasRoutePrefill.value) {
    restorePageState()
    return
  }
  savePageState()
})

watch(
  [patientId, patientName, patientAge, patientSex, threshold, previewSrc, detectionResult],
  () => {
    savePageState()
  },
  { deep: true },
)
</script>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.page-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.page-header h2 {
  margin: 0;
  color: #1a3a5c;
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-title {
  font-weight: 600;
  color: #1a3a5c;
}
.detecting-tip {
  padding: 40px;
  text-align: center;
  color: #5a7fa0;
  font-size: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
</style>
