<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Document /></el-icon> 诊断报告生成</h2>
      <el-tag>AI 辅助报告</el-tag>
    </div>

    <!-- 参数输入 -->
    <el-card shadow="never" style="margin-bottom:16px">
      <template #header><span class="card-title">报告参数</span></template>
      <el-row :gutter="24">
        <el-col :span="6">
          <el-form-item label="患者姓名" style="margin:0">
            <el-input v-model="form.patientName" placeholder="患者姓名" />
          </el-form-item>
        </el-col>
        <el-col :span="4">
          <el-form-item label="年龄" style="margin:0">
            <el-input-number v-model="form.patientAge" :min="0" style="width:100%" />
          </el-form-item>
        </el-col>
        <el-col :span="4">
          <el-form-item label="性别" style="margin:0">
            <el-select v-model="form.patientSex">
              <el-option label="男" value="男" />
              <el-option label="女" value="女" />
              <el-option label="未知" value="未知" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="6">
          <el-form-item label="影像模态" style="margin:0">
            <el-select v-model="form.modality">
              <el-option label="心脏超声（超声心动图）" value="ultrasound" />
              <el-option label="心脏 MRI（CMR）" value="mri" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="4" style="display:flex;align-items:flex-end">
          <el-button
            type="primary"
            :loading="generating"
            style="width:100%"
            @click="handleGenerate"
          >
            <el-icon><MagicStick /></el-icon> 生成报告
          </el-button>
        </el-col>
      </el-row>
    </el-card>

    <el-row :gutter="16">
      <!-- 检测结果输入 -->
      <el-col :span="8">
        <el-card shadow="never" style="height:500px;overflow-y:auto">
          <template #header><span class="card-title">检测结果</span></template>
          <el-empty v-if="!detections.length" description="暂无检测结果（将生成正常报告）" />
          <div v-for="(det, i) in detections" :key="i" class="detection-item">
            <div class="det-label">
              <el-tag :type="det.label === '正常' ? 'success' : 'danger'" size="small">
                {{ det.label }}
              </el-tag>
              <span class="det-conf">{{ (det.confidence * 100).toFixed(1) }}%</span>
            </div>
            <div v-if="det.measurements?.width_mm" class="det-measurements">
              {{ det.measurements.width_mm }}mm × {{ det.measurements.height_mm }}mm
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- 报告预览 -->
      <el-col :span="16">
        <el-card shadow="never">
          <template #header>
            <span class="card-title">报告预览</span>
            <div style="float:right;display:flex;gap:8px" v-if="reportData">
              <el-button size="small" type="success" @click="exportDocx">
                <el-icon><Download /></el-icon> 导出 Word
              </el-button>
              <el-button size="small" @click="exportText">
                <el-icon><CopyDocument /></el-icon> 复制文本
              </el-button>
            </div>
          </template>

          <div v-if="generating" class="generating-tip">
            <el-icon class="is-loading"><Loading /></el-icon>
            AI 正在生成报告，请稍候...
          </div>

          <ReportEditor
            v-else-if="reportData"
            v-model="reportData"
            :patient-info="{ name: form.patientName, age: form.patientAge, sex: form.patientSex }"
            :modality="form.modality"
          />

          <el-empty v-else description="请填写患者信息并点击「生成报告」" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { generateReport, exportReportDocx, exportReportText } from '@/api/reports.js'
import ReportEditor from '@/components/ReportEditor.vue'

const route = useRoute()

// 从路由参数初始化（从检测页跳转过来）
const form = reactive({
  patientName: route.query.name || '',
  patientAge: Number(route.query.age) || null,
  patientSex: route.query.sex || '未知',
  modality: route.query.modality || 'ultrasound',
})

const detections = ref([])
const generating = ref(false)
const reportData = ref(null)
const rawReport = ref(null)

onMounted(() => {
  if (route.query.detections) {
    try {
      detections.value = JSON.parse(route.query.detections)
    } catch {
      detections.value = []
    }
  }
})

async function handleGenerate() {
  generating.value = true
  reportData.value = null
  try {
    const payload = {
      modality: form.modality,
      patient_info: {
        name: form.patientName || '患者',
        age: form.patientAge,
        sex: form.patientSex,
      },
      detections: detections.value,
    }
    const res = await generateReport(payload)
    reportData.value = res.report_data
    rawReport.value = res
    ElMessage.success('报告生成成功')
  } finally {
    generating.value = false
  }
}

async function exportDocx() {
  try {
    const payload = {
      modality: form.modality,
      patient_info: {
        name: form.patientName || '患者',
        age: form.patientAge,
        sex: form.patientSex,
      },
      detections: detections.value,
    }
    await exportReportDocx(payload, `CHD_Report_${form.patientName || '患者'}.docx`)
    ElMessage.success('Word 报告已下载')
  } catch {
    ElMessage.error('Word 导出失败')
  }
}

async function exportText() {
  if (!reportData.value) return
  const text = [
    `【${form.modality === 'ultrasound' ? '超声心动图' : '心脏MRI（CMR）'}诊断报告】`,
    `患者：${form.patientName || '未知'} | 年龄：${form.patientAge ?? '-'}岁 | 性别：${form.patientSex}`,
    '',
    `■ 影像学表现\n${reportData.value.image_findings}`,
    `■ 异常发现\n${reportData.value.abnormal_findings}`,
    `■ 初步诊断意见\n${reportData.value.preliminary_suggestion}`,
    `■ 建议\n${reportData.value.recommendations}`,
    '',
    '【声明】本报告由 AI 辅助生成，仅供临床参考。',
  ].join('\n')

  await navigator.clipboard.writeText(text)
  ElMessage.success('报告文本已复制到剪贴板')
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
.detection-item {
  padding: 8px;
  border: 1px solid #e0eaf5;
  border-radius: 6px;
  margin-bottom: 8px;
}
.det-label {
  display: flex;
  align-items: center;
  gap: 8px;
}
.det-conf {
  font-size: 13px;
  color: #5a7fa0;
}
.det-measurements {
  font-size: 12px;
  color: #888;
  margin-top: 4px;
}
.generating-tip {
  padding: 60px;
  text-align: center;
  color: #5a7fa0;
  font-size: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
</style>
