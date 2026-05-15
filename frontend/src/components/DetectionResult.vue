<!--
  DetectionResult 组件
  展示影像检测结果：标注影像、检测列表、测量值等。
  Props:
    result: DetectionResponse（来自后端 /images/detect 接口）
    modality: 'ultrasound' | 'mri'
-->
<template>
  <div class="detection-result">
    <!-- 摘要统计 -->
    <el-row :gutter="12" style="margin-bottom:16px">
      <el-col :span="8">
        <el-statistic title="检测耗时" :value="displayResult.processing_time_s" suffix="秒" />
      </el-col>
      <el-col :span="8">
        <el-statistic title="检测项数" :value="detectionItemCount" />
      </el-col>
      <el-col :span="8">
        <el-statistic title="异常项数" :value="abnormalCount" />
      </el-col>
    </el-row>

    <!-- 3D 体数据逐层浏览：按需求临时隐藏，后续可恢复 -->

    <el-alert
      v-if="displayResult.inference_mode"
      :title="`推理模式：${displayResult.inference_mode}`"
      type="info"
      :closable="false"
      style="margin-bottom:12px"
    />

    <el-card
      v-if="modality === 'mri'"
      shadow="never"
      style="margin-bottom:12px"
    >
      <template #header>
        <span class="card-title">常模检测</span>
      </template>

      <div v-if="normalityInfo" class="normality-summary">
        <el-tag :type="normalityInfo.is_abnormal ? 'danger' : 'success'" effect="dark">
          {{ normalityInfo.is_abnormal ? '异常' : '正常' }}
        </el-tag>
      </div>
      <el-alert
        v-else
        type="warning"
        :closable="false"
        title="未获取到第二模型结果（请使用 NIfTI 3D 检测，并确认后端已加载 mri_normal_heart_mlp.pth）"
        style="margin-bottom:10px"
      />

      <div v-if="Array.isArray(normalityInfo.abnormal_features) && normalityInfo.abnormal_features.length" class="normality-abnormal">
        <p class="section-title">异常特征（Top）</p>
        <div
          v-for="(item, idx) in normalityInfo.abnormal_features.slice(0, 8)"
          :key="`${item.feature}-${idx}`"
          class="normality-abnormal-row"
        >
          <span>
            {{ formatFeatureName(item.feature) }}
            <el-tag
              v-if="getAbnormalDirection(item)"
              size="small"
              :type="getAbnormalDirection(item) === '偏大' ? 'danger' : 'warning'"
              effect="plain"
              style="margin-left:6px"
            >
              {{ getAbnormalDirection(item) }}
            </el-tag>
          </span>
          <span>
            residual={{ formatFeatureValue(item.abs_residual_std || item.abs_z) }}
            <span v-if="getNormalRangeText(item.feature, item.normal_range)">
              ｜正常范围：{{ getNormalRangeText(item.feature, item.normal_range) }}
            </span>
          </span>
        </div>
      </div>

      <el-collapse v-if="normalityInfo">
        <el-collapse-item title="分割详细数据" name="model-input-features">
          <el-descriptions :column="2" size="small" border>
            <el-descriptions-item
              v-for="item in modelInputFeatureEntries"
              :key="item.key"
              :label="formatFeatureName(item.key)"
            >
              {{ formatFeatureValue(item.value) }}
            </el-descriptions-item>
          </el-descriptions>
        </el-collapse-item>
      </el-collapse>
    </el-card>

    <el-row :gutter="12">
      <!-- 标注影像与分割展示 -->
      <el-col :span="24">
        <div class="image-grid" :class="{ 'single-panel': !displayResult.segmentation_mask_base64 }">
          <div class="annotated-image-wrap image-panel">
            <p class="section-title">标注影像</p>
            <div v-if="sliceLoading" class="slice-loading">
              <el-icon class="is-loading"><Loading /></el-icon>
              正在加载切片...
            </div>
            <img
              :src="`data:image/png;base64,${displayResult.annotated_image_base64}`"
              alt="标注影像"
              class="annotated-image"
            />
          </div>

          <div
            v-if="displayResult.segmentation_mask_base64"
            class="annotated-image-wrap image-panel"
          >
            <p class="section-title">分割 Mask</p>
            <img
              :src="`data:image/png;base64,${displayResult.segmentation_mask_base64}`"
              alt="分割掩码"
              class="annotated-image"
            />
            <div style="margin-top:8px;text-align:right">
              <el-link type="primary" @click="downloadSegmentationMask">
                下载分割 Mask
              </el-link>
            </div>

            <div v-if="showSegmentationLegend" class="seg-legend">
              <p class="section-title" style="margin-top:10px">分割图例</p>
              <div class="seg-legend-grid">
                <div
                  v-for="item in segmentationLegend"
                  :key="item.label"
                  class="seg-legend-item"
                >
                  <span class="seg-color" :style="{ backgroundColor: item.color }" />
                  <span class="seg-label">{{ item.label }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>

  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { getNiftiSlice } from '@/api/images.js'

const props = defineProps({
  result: { type: Object, required: true },
  modality: { type: String, default: 'ultrasound' },
  confidenceThreshold: { type: Number, default: 0.5 },
})

const isNifti3D = computed(() => {
  const meta = props.result?.dicom_metadata || {}
  return props.modality === 'mri' && String(meta.format || '').toLowerCase() === 'nifti'
})

const volumeDepth = computed(() => {
  const meta = props.result?.dicom_metadata || {}
  const shape = meta.nifti_shape
  return Array.isArray(shape) && shape.length > 0 ? Number(shape[0] || 0) : 0
})

const currentSlice = ref(0)
const sliceLoading = ref(false)
const sliceResult = ref(null)
let sliceRequestId = 0

watch(
  () => props.result,
  (val) => {
    if (!val) return
    // 3D逐层浏览已隐藏，直接使用接口初始返回结果展示。
    sliceResult.value = null
    if (!isNifti3D.value) return
    const meta = val.dicom_metadata || {}
    const idx = Number(meta.slice_index ?? 0)
    currentSlice.value = Number.isFinite(idx) ? idx : 0
  },
  { immediate: true },
)

async function loadSlice() {
  if (!isNifti3D.value) return
  const taskId = props.result?.task_id
  if (!taskId) return
  const depth = volumeDepth.value
  if (depth <= 0) return
  if (currentSlice.value < 0 || currentSlice.value >= depth) return

  const requestId = ++sliceRequestId
  sliceLoading.value = true
  try {
    const res = await getNiftiSlice(taskId, currentSlice.value, props.confidenceThreshold)
    if (requestId !== sliceRequestId) return
    sliceResult.value = {
      ...props.result,
      detections: res.detections || [],
      annotated_image_base64: res.annotated_image_base64,
      segmentation_mask_base64: res.segmentation_mask_base64,
      processing_time_s: res.processing_time_s,
      inference_mode: res.inference_mode,
      normality: res.normality || props.result.normality || null,
    }
  } finally {
    if (requestId === sliceRequestId) {
      sliceLoading.value = false
    }
  }
}

watch(
  () => props.confidenceThreshold,
  () => {
    // 3D逐层浏览已隐藏，暂不触发逐层重推理。
  },
)

const displayResult = computed(() => sliceResult.value || props.result)

const normalityInfo = computed(() => displayResult.value?.normality || null)
const modelInputFeatureEntries = computed(() => {
  const features = normalityInfo.value?.model_input_features || {}
  return Object.keys(features)
    .sort((a, b) => a.localeCompare(b))
    .map((key) => ({ key, value: features[key] }))
})

const detectionItemCount = computed(() => {
  if (props.modality === 'mri' && normalityInfo.value) {
    return modelInputFeatureEntries.value.length
  }
  return displayResult.value?.detections?.length || 0
})

const abnormalCount = computed(
  () => {
    if (props.modality === 'mri' && normalityInfo.value) {
      const abnormalFeatures = normalityInfo.value?.abnormal_features
      return Array.isArray(abnormalFeatures) ? abnormalFeatures.length : 0
    }
    const detections = displayResult.value?.detections || []
    return detections.filter((d) => d.label !== '正常').length
  },
)

const showSegmentationLegend = computed(
  () => props.modality === 'mri' && Boolean(displayResult.value.segmentation_mask_base64),
)

const segmentationLegend = [
  { label: '背景', color: '#000000' },
  { label: '左心室(LV)', color: '#dc2828' },
  { label: '右心室(RV)', color: '#285adc' },
  { label: '左心房(LA)', color: '#5adc28' },
  { label: '右心房(RA)', color: '#dcdc28' },
  { label: '心肌', color: '#b478dc' },
  { label: '升主动脉', color: '#dc50b4' },
  { label: '肺动脉', color: '#50b4ff' },
]

function formatFeatureValue(value) {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '-'
    return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(4)
  }
  return value
}

function formatFeatureName(name) {
  const directMap = {
    fg_total_voxels: '前景总体素数',
    fg_total_volume_ml: '前景总体积(ml)',
    ratio_lv_rv: '左室/右室体积比',
    ratio_la_ra: '左房/右房体积比',
    ratio_myo_lv: '心肌/左室体积比',
    ratio_ao_pa: '升主动脉/肺动脉体积比',
  }
  if (directMap[name]) return `${directMap[name]}（${name}）`

  const clsMap = {
    c1: '左心室',
    c2: '右心室',
    c3: '左心房',
    c4: '右心房',
    c5: '心肌',
    c6: '升主动脉',
    c7: '肺动脉',
  }
  const metricMap = {
    ratio_fg: '占前景比例',
    volume_ml: '体积(ml)',
    extent_x_mm: 'X向跨度(mm)',
    extent_y_mm: 'Y向跨度(mm)',
    extent_z_mm: 'Z向跨度(mm)',
  }
  const parts = name.split('_')
  if (parts.length >= 3 && clsMap[parts[0]]) {
    const cls = clsMap[parts[0]]
    const metric = metricMap[parts.slice(1).join('_')] || parts.slice(1).join('_')
    return `${cls}-${metric}（${name}）`
  }
  return name
}

function getAbnormalDirection(item) {
  if (!item || typeof item !== 'object') return ''
  const signed = item.residual_std ?? item.z_score
  if (typeof signed !== 'number' || !Number.isFinite(signed)) return ''
  if (signed > 0) return '偏大'
  if (signed < 0) return '偏小'
  return ''
}

function getNormalRangeText(featureName, fallbackRange) {
  const range =
    fallbackRange ||
    normalityInfo.value?.normal_ranges?.[featureName] ||
    null
  if (!range || typeof range !== 'object') return ''
  const lower = range.lower
  const upper = range.upper
  if (!Number.isFinite(lower) || !Number.isFinite(upper)) return ''
  return `${formatFeatureValue(lower)} ~ ${formatFeatureValue(upper)}`
}

function downloadSegmentationMask() {
  if (!displayResult.value.segmentation_mask_base64) return
  const a = document.createElement('a')
  a.href = `data:image/png;base64,${displayResult.value.segmentation_mask_base64}`
  a.download = 'segmentation_mask.png'
  a.click()
}
</script>

<style scoped>
.detection-result { width: 100%; }
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: #5a7fa0;
  margin: 0 0 8px;
}
.annotated-image-wrap {
  background: #f8fafd;
  border-radius: 8px;
  padding: 8px;
}
.image-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.image-grid.single-panel {
  grid-template-columns: 1fr;
}
.image-panel {
  min-height: 320px;
}
.annotated-image {
  width: 100%;
  height: 230px;
  object-fit: contain;
  background: #f0f4fa;
  border-radius: 6px;
  border: 1px solid #d0e4f5;
}
.det-card {
  padding: 10px 12px;
  border-radius: 8px;
  margin-bottom: 8px;
  border-left: 4px solid;
}
.det-normal {
  background: #f0fef4;
  border-left-color: #67c23a;
}
.det-abnormal {
  background: #fff5f5;
  border-left-color: #f56c6c;
}
.det-header {
  display: flex;
  align-items: center;
  gap: 8px;
}
.det-bbox {
  font-size: 11px;
  color: #888;
  margin-top: 4px;
}
.det-measure {
  font-size: 12px;
  color: #3a7bd5;
  margin-top: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
}
.seg-legend {
  border-top: 1px dashed #d6e4f5;
  margin-top: 10px;
  padding-top: 6px;
}
.seg-legend-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 10px;
}
.seg-legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}
.seg-color {
  width: 12px;
  height: 12px;
  border-radius: 3px;
  border: 1px solid #c8d8ea;
  flex: 0 0 12px;
}
.seg-label {
  font-size: 12px;
  color: #35526f;
}

.card-title { font-weight: 600; color: #1a3a5c; }
.slice-controls {
  display: flex;
  align-items: center;
  gap: 12px;
}
.slice-label {
  font-size: 12px;
  color: #35526f;
  min-width: 120px;
}
.slice-loading {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #5a7fa0;
  margin-bottom: 6px;
}
.normality-summary {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.normality-score {
  font-size: 12px;
  color: #35526f;
}
.normality-abnormal {
  margin-bottom: 10px;
}
.normality-abnormal-row {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #6d3d3d;
  padding: 4px 0;
  border-bottom: 1px dashed #f0d1d1;
}
</style>
