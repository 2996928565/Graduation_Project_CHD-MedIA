<template>
  <div>
    <div class="page-header">
      <h2><el-icon><PictureFilled /></el-icon> 心脏影像检测</h2>
      <div class="page-actions">
        <el-button
          v-if="isAdmin"
          type="success"
          plain
          size="small"
          @click="openNormalitySwitchDialog"
        >
          常模切换
        </el-button>
        <el-button
          v-if="isAdmin"
          type="primary"
          plain
          size="small"
          @click="openNormalityTrainDialog"
        >
          常模训练
        </el-button>
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

    <el-dialog
      v-model="trainDialogVisible"
      title="MRI 常模训练"
      width="720px"
      destroy-on-close
      @closed="stopTrainPolling"
    >
      <el-form label-width="160px">
        <el-form-item label="训练数据集（zip）">
          <el-upload
            :auto-upload="false"
            :limit="1"
            accept=".zip"
            :on-change="onTrainZipChange"
            :on-remove="onTrainZipRemove"
          >
            <el-button type="primary" plain>选择 zip</el-button>
          </el-upload>
          <div style="font-size:12px;color:#6b7a90;margin-top:6px">
            zip 内包含 *_prediction.nii.gz，可选 normal_list.txt
          </div>
        </el-form-item>
        <el-form-item label="模型名称">
          <el-input v-model="trainForm.modelName" placeholder="mri_normal_heart_mlp" />
          <div style="font-size:12px;color:#6b7a90;margin-top:6px">
            自动补全 .pth 后缀，非法字符会被过滤
          </div>
        </el-form-item>
        <el-form-item label="训练轮数">
          <el-input-number v-model="trainForm.epochs" :min="20" :max="2000" />
        </el-form-item>
        <el-form-item label="隐藏层结构">
          <el-input v-model="trainForm.hiddenDims" placeholder="64,32" />
        </el-form-item>
        <el-form-item label="批大小">
          <el-input-number v-model="trainForm.batchSize" :min="1" :max="128" />
        </el-form-item>
        <el-form-item label="学习率">
          <el-input v-model="trainForm.lr" placeholder="0.001" />
        </el-form-item>
        <el-form-item label="权重衰减">
          <el-input v-model="trainForm.weightDecay" placeholder="0.00001" />
        </el-form-item>
        <el-form-item label="阈值分位数">
          <el-input-number v-model="trainForm.thresholdQuantile" :min="0.01" :max="0.999" :step="0.001" />
        </el-form-item>
        <el-form-item label="训练设备">
          <el-select v-model="trainForm.device" style="width: 160px">
            <el-option label="cuda" value="cuda" />
            <el-option label="cpu" value="cpu" />
          </el-select>
        </el-form-item>
        <el-form-item label="正常样本清单文件名">
          <el-input v-model="trainForm.normalListName" placeholder="normal_list.txt" />
        </el-form-item>
      </el-form>

      <div style="display:flex;gap:10px;align-items:center;justify-content:flex-end;margin-top:10px">
        <el-tag v-if="trainRunId" type="info">run_id：{{ trainRunId }}</el-tag>
        <el-tag v-if="trainStatus" :type="trainStatusTagType">{{ trainStatus.status }}</el-tag>
        <el-button type="primary" :loading="trainStarting" @click="startTrain">启动训练</el-button>
      </div>

      <div v-if="trainRunId" style="margin-top:12px">
        <el-input
          v-model="trainLog"
          type="textarea"
          :rows="12"
          readonly
          placeholder="训练日志将在此显示"
        />
      </div>
    </el-dialog>

    <el-dialog
      v-model="switchDialogVisible"
      title="常模模型切换"
      width="620px"
      destroy-on-close
    >
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div style="color:#1a3a5c;font-weight:600">选择要启用的常模模型</div>
        <div style="font-size:12px;color:#6b7a90">当前：{{ activeModelLabel }}</div>
      </div>
      <div style="display:flex;gap:10px;align-items:center;margin-top:14px">
        <el-select
          v-model="selectedModelId"
          placeholder="选择要启用的模型"
          filterable
          style="flex: 1"
          :loading="normalityModelsLoading"
        >
          <el-option
            v-for="m in normalityModels"
            :key="m.id"
            :label="`${m.display_name || '未命名'}${m.is_active ? '（已启用）' : ''}`"
            :value="m.id"
          />
        </el-select>
        <el-button :loading="normalityModelsLoading" @click="fetchNormalityModels">刷新</el-button>
        <el-button type="success" plain @click="activateSelectedModel">启用</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { detectImage, startMriNormalityTrainMlp, getMriNormalityTrainMlpStatus, getMriNormalityTrainMlpLog, listNormalityModels, activateNormalityModel } from '@/api/images.js'
import ImageUpload from '@/components/ImageUpload.vue'
import DetectionResult from '@/components/DetectionResult.vue'
import { useAuthStore } from '@/store/auth.js'

const route = useRoute()
const router = useRouter()
const CACHE_KEY = 'chd_mri_detection_state_v1'
const authStore = useAuthStore()
const isAdmin = computed(() => (authStore.role || '').toLowerCase() === 'admin')

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

const trainDialogVisible = ref(false)
const switchDialogVisible = ref(false)
const trainZipFile = ref(null)
const trainStarting = ref(false)
const trainRunId = ref('')
const trainStatus = ref(null)
const trainLog = ref('')
const trainPollTimer = ref(null)
const trainForm = ref({
  modelName: 'mri_normal_heart_mlp',
  epochs: 200,
  hiddenDims: '64,32',
  latentDim: 8,
  batchSize: 8,
  lr: '0.001',
  weightDecay: '0.00001',
  thresholdQuantile: 0.99,
  device: 'cuda',
  predIsRawMmwhs: false,
  normalListName: 'normal_list.txt',
})

const trainStatusTagType = computed(() => {
  const s = (trainStatus.value?.status || '').toLowerCase()
  if (s === 'running' || s === 'starting') return 'warning'
  if (s === 'succeeded') return 'success'
  if (s === 'failed') return 'danger'
  return 'info'
})

function openNormalityTrainDialog() {
  trainDialogVisible.value = true
}

function openNormalitySwitchDialog() {
  switchDialogVisible.value = true
  fetchNormalityModels()
}

function onTrainZipChange(file) {
  trainZipFile.value = file?.raw || null
}

function onTrainZipRemove() {
  trainZipFile.value = null
}

async function startTrain() {
  if (!trainZipFile.value) {
    ElMessage.warning('请先选择 zip 数据集')
    return
  }
  trainStarting.value = true
  try {
    const res = await startMriNormalityTrainMlp(trainZipFile.value, {
      model_name: trainForm.value.modelName,
      epochs: trainForm.value.epochs,
      hidden_dims: trainForm.value.hiddenDims,
      latent_dim: trainForm.value.latentDim,
      batch_size: trainForm.value.batchSize,
      lr: Number(trainForm.value.lr),
      weight_decay: Number(trainForm.value.weightDecay),
      threshold_quantile: trainForm.value.thresholdQuantile,
      device: trainForm.value.device,
      pred_is_raw_mmwhs: trainForm.value.predIsRawMmwhs,
      normal_list_name: trainForm.value.normalListName,
    })
    trainRunId.value = res.run_id
    trainStatus.value = { status: res.status, started_at: res.started_at }
    trainLog.value = ''
    ElMessage.success('训练任务已启动')
    startTrainPolling()
  } finally {
    trainStarting.value = false
  }
}

async function pollTrainOnce() {
  if (!trainRunId.value) return
  try {
    const s = await getMriNormalityTrainMlpStatus(trainRunId.value)
    trainStatus.value = s
  } catch {
    return
  }
  try {
    const l = await getMriNormalityTrainMlpLog(trainRunId.value, 300)
    trainLog.value = l.log || ''
  } catch {
    return
  }
}

function startTrainPolling() {
  stopTrainPolling()
  pollTrainOnce()
  trainPollTimer.value = setInterval(() => {
    pollTrainOnce()
  }, 2000)
}

function stopTrainPolling() {
  if (trainPollTimer.value) {
    clearInterval(trainPollTimer.value)
    trainPollTimer.value = null
  }
}

const normalityModelsLoading = ref(false)
const normalityModels = ref([])
const selectedModelId = ref(null)

const activeModelLabel = computed(() => {
  const active = (normalityModels.value || []).find((m) => m.is_active)
  return active ? `${active.display_name || '未命名'}（已启用）` : '未启用'
})

async function fetchNormalityModels() {
  if (!isAdmin.value) return
  normalityModelsLoading.value = true
  try {
    const rows = await listNormalityModels('mri')
    normalityModels.value = Array.isArray(rows) ? rows : []
    const active = normalityModels.value.find((m) => m.is_active)
    selectedModelId.value = active ? active.id : (normalityModels.value[0]?.id ?? null)
  } finally {
    normalityModelsLoading.value = false
  }
}

async function activateSelectedModel() {
  if (!selectedModelId.value) {
    ElMessage.warning('请选择要启用的模型')
    return
  }
  try {
    await activateNormalityModel(selectedModelId.value)
    ElMessage.success('已切换常模模型')
    fetchNormalityModels()
  } catch {
    return
  }
}

onBeforeUnmount(() => {
  stopTrainPolling()
})
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
