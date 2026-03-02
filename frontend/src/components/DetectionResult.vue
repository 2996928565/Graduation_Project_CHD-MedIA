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
        <el-statistic title="检测耗时" :value="result.processing_time_s" suffix="秒" />
      </el-col>
      <el-col :span="8">
        <el-statistic title="检测项数" :value="result.detections.length" />
      </el-col>
      <el-col :span="8">
        <el-statistic title="异常项数" :value="abnormalCount" />
      </el-col>
    </el-row>

    <el-row :gutter="12">
      <!-- 标注影像 -->
      <el-col :span="12">
        <div class="annotated-image-wrap">
          <p class="section-title">标注影像</p>
          <img
            :src="`data:image/png;base64,${result.annotated_image_base64}`"
            alt="标注影像"
            class="annotated-image"
          />
        </div>
      </el-col>

      <!-- 检测列表 -->
      <el-col :span="12">
        <p class="section-title">检测结果列表</p>
        <div v-if="result.detections.length === 0">
          <el-empty description="未检测到异常" :image-size="60" />
        </div>
        <div
          v-for="(det, i) in result.detections"
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
import { computed } from 'vue'

const props = defineProps({
  result: { type: Object, required: true },
  modality: { type: String, default: 'ultrasound' },
})

const abnormalCount = computed(
  () => props.result.detections.filter((d) => d.label !== '正常').length,
)
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
</style>
