<template>
  <div>
    <div class="page-header">
      <h2><el-icon><PictureFilled /></el-icon> 心脏影像检测</h2>
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
        <el-tag type="warning">心脏磁共振成像（CMR）</el-tag>
      </div>
    </div>

    <!-- 患者信息 -->
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
      </el-row>
    </el-card>

    <el-row :gutter="16">
      <el-col :span="5">
        <el-card shadow="never">
          <template #header><span class="card-title">上传 MRI 影像</span></template>
          <ImageUpload
            ref="uploadRef"
            modality="mri"
            accept=".png,.jpg,.jpeg,.dcm,.dicom,.nii,.nii.gz"
            @file-selected="onFileSelected"
          />
          <div v-if="previewSrc" style="margin-top:12px">
            <p style="color:#5a7fa0;font-size:13px;margin:0 0 8px">原始影像预览：</p>
            <img :src="previewSrc" style="width:100%;max-height:180px;object-fit:contain;border-radius:8px;border:1px solid #e0eaf5" />
          </div>
        </el-card>
      </el-col>

      <el-col :span="19">
        <el-card shadow="never">
          <template #header>
            <span class="card-title">检测结果</span>
            <el-button
              v-if="selectedFile"
              type="warning"
              size="small"
              :loading="detecting"
              style="float:right"
              @click="runDetection"
            >
              <el-icon><Search /></el-icon> 开始影像分析
            </el-button>
          </template>

          <div v-if="detecting" class="detecting-tip">
            <el-icon class="is-loading"><Loading /></el-icon>
            正在分析影像，U-Net 分割中，请稍候...
          </div>

          <DetectionResult
            v-else-if="detectionResult"
            :result="detectionResult"
            modality="mri"
            :confidence-threshold="0.5"
          />

          <el-empty v-else description="请上传心脏影像（仅支持 NIfTI）" />
        </el-card>
      </el-col>
    </el-row>

    <div v-if="detectionResult" style="margin-top:16px;text-align:right">
      <el-button type="success" icon="Document" @click="goToReport">
        生成诊断报告
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { detectImage } from '@/api/images.js'
import ImageUpload from '@/components/ImageUpload.vue'
import DetectionResult from '@/components/DetectionResult.vue'

const route = useRoute()
const router = useRouter()
const CACHE_KEY = 'chd_mri_detection_state_v1'

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

const uploadRef = ref(null)
const selectedFile = ref(null)
const previewSrc = ref('')
const dicomMeta = ref(null)
const detecting = ref(false)
const detectionResult = ref(null)
const detectRequestId = ref(0)
const hasRoutePrefill = computed(
  () => Boolean(patientId.value || route.query.name || route.query.age || route.query.sex),
)
const hasContent = computed(
  () => Boolean(
    selectedFile.value ||
    previewSrc.value ||
    dicomMeta.value ||
    detectionResult.value ||
    patientId.value ||
    patientName.value ||
    patientAge.value !== null ||
    (patientSex.value && patientSex.value !== '未知')
  ),
)

const isNiftiFile = computed(() => {
  const name = (selectedFile.value?.name || '').toLowerCase()
  return name.endsWith('.nii.gz') || name.endsWith('.nii')
})

function onFileSelected({ file, previewBase64, metadata }) {
  selectedFile.value = file
  previewSrc.value = !file
    ? ''
    : previewBase64
      ? `data:image/png;base64,${previewBase64}`
      : URL.createObjectURL(file)
  dicomMeta.value = metadata || null
  detectionResult.value = null

  if (file) {
    runDetection({ silent: true })
  }
}

function clearCurrentState() {
  selectedFile.value = null
  previewSrc.value = ''
  dicomMeta.value = null
  detectionResult.value = null
  detecting.value = false
  detectRequestId.value += 1
  patientId.value = ''
  patientName.value = ''
  patientAge.value = null
  patientSex.value = '未知'
  uploadRef.value?.clearFile?.()
  sessionStorage.removeItem(CACHE_KEY)
  sessionStorage.removeItem('chd_report_source_state_v1')
  router.replace({ path: route.path, query: {} })
  ElMessage.success('已清空当前检测内容')
}

async function runDetection({ silent = false } = {}) {
  if (!selectedFile.value) {
    if (!silent) ElMessage.warning('请先上传影像')
    return
  }

  const requestId = ++detectRequestId.value
  detecting.value = true
  try {
    const result = await detectImage(
      selectedFile.value,
      'mri',
      0.5,
      patientId.value || null,
    )
    if (requestId !== detectRequestId.value) {
      return
    }
    detectionResult.value = result
    if (!silent) {
      ElMessage.success(`影像分析完成，发现 ${result.detections.length} 条结果`)
    }
  } catch {
    if (!silent) {
      ElMessage.error('影像分析失败，请重试')
    }
  } finally {
    if (requestId === detectRequestId.value) {
      detecting.value = false
    }
  }
}

function goToReport() {
  sessionStorage.setItem(
    'chd_report_source_state_v1',
    JSON.stringify({
      modality: 'mri',
      patientId: patientId.value || '',
      name: patientName.value,
      age: patientAge.value,
      sex: patientSex.value,
      detections: detectionResult.value?.detections || [],
      normality: detectionResult.value?.normality || null,
      annotated_image_base64: detectionResult.value?.annotated_image_base64 || '',
      segmentation_mask_base64: detectionResult.value?.segmentation_mask_base64 || '',
    }),
  )
  router.push({
    path: '/report',
    query: {
      modality: 'mri',
      patientId: patientId.value || '',
      name: patientName.value,
      age: patientAge.value,
      sex: patientSex.value,
      detections: JSON.stringify(detectionResult.value?.detections || []),
      normality: JSON.stringify(detectionResult.value?.normality || null),
    },
  })
}

function savePageState() {
  const payload = {
    patientId: patientId.value || '',
    patientName: patientName.value || '',
    patientAge: patientAge.value,
    patientSex: patientSex.value || '未知',
    previewSrc: previewSrc.value || '',
    dicomMeta: dicomMeta.value || null,
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
    previewSrc.value = state.previewSrc || ''
    dicomMeta.value = state.dicomMeta || null
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
  [patientId, patientName, patientAge, patientSex, previewSrc, dicomMeta, detectionResult],
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
.card-title { font-weight: 600; color: #1a3a5c; }
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
