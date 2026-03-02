<template>
  <div>
    <div class="page-header">
      <h2><el-icon><VideoCamera /></el-icon> 心脏超声检测</h2>
      <el-tag type="primary">超声心动图 / 二维超声</el-tag>
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
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { detectImage, uploadPreview } from '@/api/images.js'
import ImageUpload from '@/components/ImageUpload.vue'
import DetectionResult from '@/components/DetectionResult.vue'

const route = useRoute()
const router = useRouter()

const patientName = ref(route.query.name || '')
const patientAge = ref(null)
const patientSex = ref('未知')
const threshold = ref(0.5)

const selectedFile = ref(null)
const previewSrc = ref('')
const detecting = ref(false)
const detectionResult = ref(null)

function onFileSelected({ file, previewBase64 }) {
  selectedFile.value = file
  previewSrc.value = previewBase64
    ? `data:image/png;base64,${previewBase64}`
    : URL.createObjectURL(file)
  detectionResult.value = null
}

async function runDetection() {
  if (!selectedFile.value) {
    ElMessage.warning('请先上传超声影像')
    return
  }
  detecting.value = true
  try {
    const result = await detectImage(selectedFile.value, 'ultrasound', threshold.value)
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
