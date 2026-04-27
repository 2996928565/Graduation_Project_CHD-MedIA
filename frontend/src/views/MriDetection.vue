<template>
  <div>
    <div class="page-header">
      <h2><el-icon><PictureFilled /></el-icon> 心脏影像检测</h2>
      <el-tag type="warning">心脏磁共振成像（CMR）</el-tag>
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
        <el-col :span="4">
          <el-form-item label="置信度阈值" style="margin:0">
            <el-slider v-model="threshold" :min="0.1" :max="0.9" :step="0.05" :format-tooltip="v => (v*100).toFixed(0)+'%'" />
          </el-form-item>
        </el-col>
      </el-row>
    </el-card>

    <el-row :gutter="16">
      <el-col :span="10">
        <el-card shadow="never">
          <template #header><span class="card-title">上传 MRI 影像</span></template>
          <ImageUpload
            modality="mri"
            accept=".png,.jpg,.jpeg,.dcm,.dicom,.nii,.nii.gz"
            @file-selected="onFileSelected"
          />
          <div v-if="previewSrc" style="margin-top:12px">
            <p style="color:#5a7fa0;font-size:13px;margin:0 0 8px">原始影像预览：</p>
            <img :src="previewSrc" style="max-width:100%;border-radius:8px;border:1px solid #e0eaf5" />
          </div>
          <!-- DICOM 元数据展示 -->
          <div v-if="dicomMeta && Object.keys(dicomMeta).length" style="margin-top:12px">
            <el-descriptions title="DICOM 元数据" :column="2" size="small" border>
              <el-descriptions-item
                v-for="(val, key) in filteredMeta"
                :key="key"
                :label="key"
              >{{ val || '-' }}</el-descriptions-item>
            </el-descriptions>
          </div>
        </el-card>
      </el-col>

      <el-col :span="14">
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
            :confidence-threshold="threshold"
          />

          <el-empty v-else description="请上传心脏影像（支持 NIfTI/DICOM/PNG/JPG）" />
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
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { detectImage } from '@/api/images.js'
import ImageUpload from '@/components/ImageUpload.vue'
import DetectionResult from '@/components/DetectionResult.vue'

const route = useRoute()
const router = useRouter()

function parseQueryAge(v) {
  if (v === undefined || v === null || v === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

function normalizeSex(v) {
  return ['男', '女', '未知'].includes(v) ? v : '未知'
}

const patientName = ref(route.query.name || '')
const patientAge = ref(parseQueryAge(route.query.age))
const patientSex = ref(normalizeSex(route.query.sex))
const threshold = ref(0.5)

const selectedFile = ref(null)
const previewSrc = ref('')
const dicomMeta = ref(null)
const detecting = ref(false)
const detectionResult = ref(null)
const detectRequestId = ref(0)

const isNiftiFile = computed(() => {
  const name = (selectedFile.value?.name || '').toLowerCase()
  return name.endsWith('.nii.gz') || name.endsWith('.nii')
})

const filteredMeta = computed(() => {
  if (!dicomMeta.value) return {}
  const keys = ['modality', 'patient_name', 'patient_age', 'patient_sex', 'study_date', 'series_description', 'rows', 'columns']
  return Object.fromEntries(keys.map(k => [k, dicomMeta.value[k]]).filter(([, v]) => v))
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
      threshold.value,
      route.query.patientId || null,
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
  router.push({
    path: '/report',
    query: {
      modality: 'mri',
      patientId: route.query.patientId || '',
      name: patientName.value,
      age: patientAge.value,
      sex: patientSex.value,
      detections: JSON.stringify(detectionResult.value?.detections || []),
    },
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
