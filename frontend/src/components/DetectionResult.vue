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
        <el-statistic title="检测项数" :value="displayResult.detections.length" />
      </el-col>
      <el-col :span="8">
        <el-statistic title="异常项数" :value="abnormalCount" />
      </el-col>
    </el-row>

    <el-card
      v-if="isNifti3D"
      shadow="never"
      style="margin-bottom:12px"
    >
      <template #header>
        <span class="card-title">3D 体数据逐层浏览</span>
      </template>
      <div class="slice-controls">
        <div class="slice-label">切片：{{ currentSlice + 1 }} / {{ volumeDepth }}</div>
        <el-slider
          v-model="currentSlice"
          :min="0"
          :max="Math.max(volumeDepth - 1, 0)"
          :step="1"
          :show-tooltip="false"
          @change="loadSlice"
        />
      </div>
    </el-card>

    <el-alert
      v-if="displayResult.inference_mode"
      :title="`推理模式：${displayResult.inference_mode}`"
      type="info"
      :closable="false"
      style="margin-bottom:12px"
    />

    <el-row :gutter="12">
      <!-- 标注影像 -->
      <el-col :span="12">
        <div class="annotated-image-wrap">
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
          class="annotated-image-wrap"
          style="margin-top:10px"
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
      </el-col>

      <!-- 检测列表 -->
      <el-col :span="12">
        <p class="section-title">检测结果列表</p>
        <div v-if="displayResult.detections.length === 0">
          <el-empty description="未检测到异常" :image-size="60" />
        </div>
        <div
          v-for="(det, i) in displayResult.detections"
          :key="i"
          class="det-card"
          :class="{ 'det-normal': det.label === '正常', 'det-abnormal': det.label !== '正常' }"
        >
          <div class="det-header">
            <el-tag
              :type="det.label === '正常' ? 'success' : 'danger'"
              size="small"
              effect="dark"
            >{{ det.label }}</el-tag>
            <el-progress
              :percentage="Math.round(det.confidence * 100)"
              :color="det.label === '正常' ? '#67c23a' : '#f56c6c'"
              :stroke-width="8"
              style="width:120px;margin-left:auto"
            />
          </div>

          <div v-if="det.bbox && det.bbox.length === 4" class="det-bbox">
            位置：({{ det.bbox.map(v => Math.round(v)).join(', ') }})
          </div>

          <div v-if="det.measurements && det.measurements.width_mm" class="det-measure">
            <el-icon><Ruler /></el-icon>
            {{ det.measurements.width_mm }}mm × {{ det.measurements.height_mm }}mm
            <span v-if="det.measurements.area_mm2">
              （面积 {{ det.measurements.area_mm2 }} mm²）
            </span>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- DICOM 元数据 -->
    <div v-if="result.dicom_metadata && Object.keys(result.dicom_metadata).length" style="margin-top:12px">
      <el-collapse>
        <el-collapse-item title="DICOM 元数据" name="dicom">
          <el-descriptions :column="3" size="small" border>
            <el-descriptions-item
              v-for="(val, key) in result.dicom_metadata"
              :key="key"
              :label="key"
            >{{ val || '-' }}</el-descriptions-item>
          </el-descriptions>
        </el-collapse-item>
      </el-collapse>
    </div>
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
    if (!isNifti3D.value) {
      sliceResult.value = null
      return
    }
    const meta = val.dicom_metadata || {}
    const idx = Number(meta.slice_index ?? 0)
    currentSlice.value = Number.isFinite(idx) ? idx : 0
    loadSlice(currentSlice.value)
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
    if (isNifti3D.value) {
      loadSlice()
    }
  },
)

const displayResult = computed(() => sliceResult.value || props.result)

const abnormalCount = computed(
  () => displayResult.value.detections.filter((d) => d.label !== '正常').length,
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
.annotated-image {
  width: 100%;
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
</style>
