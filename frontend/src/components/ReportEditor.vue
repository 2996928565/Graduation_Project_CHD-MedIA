<!--
  ReportEditor 组件
  显示并支持编辑 AI 生成的诊断报告内容。
  Props:
    modelValue: 报告数据对象 { exam_type, exam_part, image_findings, ... }
    patientInfo: { name, age, sex }
    modality: 'ultrasound' | 'mri'
  Emits:
    update:modelValue
-->
<template>
  <div class="report-editor">
    <!-- 报告头部 -->
    <div class="report-header">
      <h3>先天性心脏病影像诊断报告（初步）</h3>
      <el-tag type="warning" size="small">AI 辅助生成 · 仅供临床参考</el-tag>
    </div>

    <el-descriptions :column="4" size="small" border style="margin-bottom:16px">
      <el-descriptions-item label="患者姓名">{{ patientInfo.name || '-' }}</el-descriptions-item>
      <el-descriptions-item label="年龄">{{ patientInfo.age != null ? patientInfo.age + '岁' : '-' }}</el-descriptions-item>
      <el-descriptions-item label="性别">{{ patientInfo.sex || '-' }}</el-descriptions-item>
      <el-descriptions-item label="检查类型">{{ modelValue.exam_type }}</el-descriptions-item>
      <el-descriptions-item label="检查部位">{{ modelValue.exam_part }}</el-descriptions-item>
      <el-descriptions-item label="检查日期" :span="3">{{ today }}</el-descriptions-item>
    </el-descriptions>

    <!-- 报告各节 -->
    <div v-for="section in sections" :key="section.key" class="report-section">
      <div class="section-label">{{ section.label }}</div>
      <el-input
        v-model="localReport[section.key]"
        type="textarea"
        :rows="section.rows || 3"
        :placeholder="`请填写${section.label}`"
        @input="emitUpdate"
      />
    </div>

    <!-- 免责声明 -->
    <div class="disclaimer">
      <el-icon><Warning /></el-icon>
      本报告由 AI 辅助生成，仅供临床参考，不作为最终诊断依据。最终诊断请以临床医师诊断为准。
    </div>
  </div>
</template>

<script setup>
import { reactive, watch, computed } from 'vue'

const props = defineProps({
  modelValue: { type: Object, required: true },
  patientInfo: { type: Object, default: () => ({}) },
  modality: { type: String, default: 'ultrasound' },
})

const emit = defineEmits(['update:modelValue'])

const today = new Date().toLocaleDateString('zh-CN', {
  year: 'numeric', month: 'long', day: 'numeric',
})

const localReport = reactive({ ...props.modelValue })

watch(() => props.modelValue, (val) => {
  Object.assign(localReport, val)
}, { deep: true })

const sections = [
  { key: 'image_findings', label: '一、影像学表现', rows: 4 },
  { key: 'abnormal_findings', label: '二、异常发现', rows: 3 },
  { key: 'preliminary_suggestion', label: '三、初步诊断意见', rows: 3 },
  { key: 'recommendations', label: '四、建议', rows: 2 },
]

function emitUpdate() {
  emit('update:modelValue', { ...localReport })
}
</script>

<style scoped>
.report-editor {
  padding: 4px;
}
.report-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 2px solid #e0eaf5;
}
.report-header h3 {
  margin: 0;
  color: #1a3a5c;
  font-size: 16px;
}
.report-section {
  margin-bottom: 16px;
}
.section-label {
  font-size: 14px;
  font-weight: 600;
  color: #2c5f8a;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
}
.section-label::before {
  content: '';
  display: inline-block;
  width: 3px;
  height: 14px;
  background: #3a7bd5;
  border-radius: 2px;
}
.disclaimer {
  margin-top: 16px;
  padding: 10px 12px;
  background: #fffbf0;
  border: 1px solid #ffe9a0;
  border-radius: 6px;
  font-size: 12px;
  color: #9a7010;
  display: flex;
  align-items: flex-start;
  gap: 6px;
}
</style>
