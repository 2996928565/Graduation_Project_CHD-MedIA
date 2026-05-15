<template>
  <div>
    <div class="page-header">
      <h2><el-icon><Document /></el-icon> 诊断报告生成</h2>
      <el-tag>结构化模板报告</el-tag>
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
        <el-col :span="5">
          <el-form-item label="影像模态" style="margin:0">
            <el-select v-model="form.modality">
              <el-option label="心脏超声（超声心动图）" value="ultrasound" />
              <el-option label="心脏 MRI（CMR）" value="mri" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :span="5" style="display:flex;align-items:flex-end">
          <div class="report-action-row">
            <el-button class="import-btn" type="primary" size="small" @click="openHistoryImport">
              从历史导入
            </el-button>
            <el-button
              type="primary"
              size="small"
              :loading="generating"
              @click="handleGenerate"
            >
              <el-icon><MagicStick /></el-icon> 生成报告
            </el-button>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <el-row :gutter="16">
      <!-- 检测结果输入 -->
      <el-col :span="8">
        <el-card shadow="never" style="height:500px;overflow-y:auto">
          <template #header><span class="card-title">第二模型检测数据</span></template>
          <el-empty
            v-if="!displayFeatureRows.length"
            description="暂无第二模型数据（将按默认内容生成报告）"
          />

          <div
            v-for="(row, i) in displayFeatureRows"
            :key="`${row.name}-${i}`"
            class="detection-item"
            :class="{ 'feature-abnormal': row.isAbnormal, 'feature-normal': !row.isAbnormal }"
          >
            <div class="det-label">
              <el-tag :type="row.isAbnormal ? 'danger' : 'success'" size="small">
                {{ row.isAbnormal ? '异常' : '正常' }}
              </el-tag>
              <span class="det-conf">{{ formatFeatureName(row.name) }}</span>
            </div>
            <div class="det-measurements">
              值：{{ formatFeatureValue(row.value) }}
              <span v-if="row.direction" style="margin-left:8px">
                （{{ row.direction }}）
              </span>
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
            正在生成报告，请稍候...
          </div>

          <template v-else-if="reportData">
            <el-row v-if="reportImages.annotated_image_base64 || reportImages.segmentation_mask_base64" :gutter="12" style="margin-bottom:12px">
              <el-col :span="12" v-if="reportImages.annotated_image_base64">
                <div class="report-image-wrap">
                  <p class="report-image-title">标注影像</p>
                  <img :src="`data:image/png;base64,${reportImages.annotated_image_base64}`" alt="标注影像" class="report-image" />
                </div>
              </el-col>
              <el-col :span="12" v-if="reportImages.segmentation_mask_base64">
                <div class="report-image-wrap">
                  <p class="report-image-title">分割 Mask</p>
                  <img :src="`data:image/png;base64,${reportImages.segmentation_mask_base64}`" alt="分割掩码" class="report-image" />
                </div>
              </el-col>
            </el-row>

            <ReportEditor
              v-model="reportData"
              :patient-info="{ name: form.patientName, age: form.patientAge, sex: form.patientSex }"
              :modality="form.modality"
            />
          </template>

          <el-empty v-else description="请填写患者信息并点击「生成报告」" />
        </el-card>
      </el-col>
    </el-row>

    <el-dialog
      v-model="historyDialogVisible"
      title="从历史记录导入"
      width="72%"
      destroy-on-close
    >
      <el-form :inline="true" @submit.prevent>
        <el-form-item label="患者姓名">
          <el-input
            v-model="historyFilter.patient_name"
            placeholder="输入患者姓名筛选"
            clearable
            @keyup.enter="fetchHistoryForImport"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="fetchHistoryForImport">搜索</el-button>
          <el-button @click="resetHistoryFilter">重置</el-button>
        </el-form-item>
      </el-form>

      <el-table
        v-loading="historyLoading"
        :data="historyRows"
        stripe
        style="width:100%"
        empty-text="暂无可导入的历史记录"
      >
        <el-table-column prop="patient_name" label="患者姓名" width="120" />
        <el-table-column prop="modality" label="模态" width="90">
          <template #default="{ row }">
            {{ row.modality === 'mri' ? '影像' : '超声' }}
          </template>
        </el-table-column>
        <el-table-column prop="filename" label="文件名" min-width="200" show-overflow-tooltip />
        <el-table-column prop="created_at" label="检测时间" width="180">
          <template #default="{ row }">{{ formatDate(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button type="primary" link @click="importHistoryRecord(row.task_id)">导入</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { generateReport, exportReportDocx, exportReportText } from '@/api/reports.js'
import { getDetectionHistory, getDetectionHistoryDetail } from '@/api/images.js'
import ReportEditor from '@/components/ReportEditor.vue'

const route = useRoute()

// 从路由参数初始化（从检测页跳转过来）
const form = reactive({
  patientName: route.query.name || '',
  patientAge: Number(route.query.age) || null,
  patientSex: route.query.sex || '未知',
  modality: route.query.modality || 'ultrasound',
  patientId: route.query.patientId || '',
})

const detections = ref([])
const normality = ref(null)
const reportImages = reactive({
  annotated_image_base64: '',
  segmentation_mask_base64: '',
})
const generating = ref(false)
const reportData = ref(null)
const rawReport = ref(null)
const historyDialogVisible = ref(false)
const historyLoading = ref(false)
const historyRows = ref([])
const historyFilter = reactive({
  patient_name: '',
})

onMounted(() => {
  if (route.query.detections) {
    try {
      detections.value = JSON.parse(route.query.detections)
    } catch {
      detections.value = []
    }
  }
  if (route.query.normality) {
    try {
      normality.value = JSON.parse(route.query.normality)
    } catch {
      normality.value = null
    }
  }
  restoreReportSourceState()
})

const abnormalFeatureMap = computed(() => {
  const rows = normality.value?.abnormal_features || []
  const out = new Map()
  rows.forEach((item) => {
    const key = item?.feature
    if (!key) return
    const signed = item?.residual_std ?? item?.z_score
    let direction = ''
    if (typeof signed === 'number') {
      if (signed > 0) direction = '偏高'
      else if (signed < 0) direction = '偏低'
    }
    out.set(key, { signed, direction })
  })
  return out
})

const displayFeatureRows = computed(() => {
  const features = normality.value?.model_input_features || {}
  const rows = Object.keys(features).map((name) => {
    const abnormal = abnormalFeatureMap.value.get(name)
    return {
      name,
      value: features[name],
      isAbnormal: Boolean(abnormal),
      direction: abnormal?.direction || '',
    }
  })

  rows.sort((a, b) => {
    if (a.isAbnormal !== b.isAbnormal) return a.isAbnormal ? -1 : 1
    return a.name.localeCompare(b.name)
  })
  return rows
})

async function handleGenerate() {
  generating.value = true
  reportData.value = null
  try {
    const payload = {
      modality: form.modality,
      patient_info: {
        patient_id: form.patientId || null,
        name: form.patientName || '患者',
        age: form.patientAge,
        sex: form.patientSex,
      },
      detections: detections.value,
      normality: normality.value,
      report_images: {
        annotated_image_base64: reportImages.annotated_image_base64 || null,
        segmentation_mask_base64: reportImages.segmentation_mask_base64 || null,
      },
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
        patient_id: form.patientId || null,
        name: form.patientName || '患者',
        age: form.patientAge,
        sex: form.patientSex,
      },
      detections: detections.value,
      normality: normality.value,
      report_images: {
        annotated_image_base64: reportImages.annotated_image_base64 || null,
        segmentation_mask_base64: reportImages.segmentation_mask_base64 || null,
      },
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
    `■ 全部检测结果\n${reportData.value.all_detections}`,
    `■ 异常发现\n${reportData.value.abnormal_findings}`,
    `■ 初步诊断意见\n${reportData.value.preliminary_suggestion}`,
    `■ 建议\n${reportData.value.recommendations}`,
    '',
    '【声明】本报告由系统根据检测结果按固定模板自动生成，仅供临床参考。',
  ].join('\n')

  await navigator.clipboard.writeText(text)
  ElMessage.success('报告文本已复制到剪贴板')
}

function formatFeatureValue(value) {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '-'
    return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(4)
  }
  return String(value)
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

function openHistoryImport() {
  historyDialogVisible.value = true
  fetchHistoryForImport()
}

async function fetchHistoryForImport() {
  historyLoading.value = true
  try {
    const res = await getDetectionHistory({
      page: 1,
      page_size: 50,
      patient_name: (historyFilter.patient_name || '').trim(),
      modality: 'mri',
    })
    historyRows.value = res.items || []
  } finally {
    historyLoading.value = false
  }
}

function resetHistoryFilter() {
  historyFilter.patient_name = ''
  fetchHistoryForImport()
}

async function importHistoryRecord(taskId) {
  if (!taskId) return
  historyLoading.value = true
  try {
    const detail = await getDetectionHistoryDetail(taskId)
    form.patientId = detail.patient_id || ''
    form.patientName = detail.patient_name || ''
    form.modality = detail.modality || 'mri'
    detections.value = Array.isArray(detail.detections) ? detail.detections : []
    normality.value = detail.normality || null
    reportImages.annotated_image_base64 = detail.annotated_image_base64 || ''
    reportImages.segmentation_mask_base64 = detail.segmentation_mask_base64 || ''

    const ageMeta = Number(detail?.dicom_metadata?.patient_age)
    form.patientAge = Number.isFinite(ageMeta) ? ageMeta : form.patientAge
    form.patientSex = detail?.dicom_metadata?.patient_sex || form.patientSex

    historyDialogVisible.value = false
    ElMessage.success('历史记录导入成功')
  } catch {
    ElMessage.error('导入历史记录失败')
  } finally {
    historyLoading.value = false
  }
}

function restoreReportSourceState() {
  const raw = sessionStorage.getItem('chd_report_source_state_v1')
  if (!raw) return
  try {
    const state = JSON.parse(raw)
    if (state?.modality) form.modality = state.modality
    if (state?.patientId) form.patientId = state.patientId
    if (state?.name) form.patientName = state.name
    if (state?.age !== undefined && state?.age !== null && state?.age !== '') {
      const age = Number(state.age)
      if (Number.isFinite(age)) form.patientAge = age
    }
    if (state?.sex) form.patientSex = state.sex
    if (Array.isArray(state?.detections)) detections.value = state.detections
    if (state?.normality) normality.value = state.normality
    reportImages.annotated_image_base64 = state?.annotated_image_base64 || reportImages.annotated_image_base64
    reportImages.segmentation_mask_base64 = state?.segmentation_mask_base64 || reportImages.segmentation_mask_base64
  } catch {
    // ignore invalid cache
  }
}

function formatDate(iso) {
  return iso ? iso.replace('T', ' ').slice(0, 19) : ''
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
.feature-abnormal {
  background: #fff2f2;
  border-color: #f6c5c5;
}
.feature-normal {
  background: #f6fbff;
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
.report-image-wrap {
  background: #f8fafd;
  border: 1px solid #dbe7f5;
  border-radius: 8px;
  padding: 8px;
}
.report-image-title {
  margin: 0 0 6px;
  color: #5a7fa0;
  font-size: 12px;
}
.report-image {
  width: 100%;
  height: 180px;
  object-fit: contain;
  background: #edf3fa;
  border-radius: 6px;
}
.report-action-row {
  width: 100%;
  display: flex;
  gap: 6px;
  justify-content: flex-end;
}
.import-btn {
  background-color: #1f5fbf;
  border-color: #1f5fbf;
}
.import-btn:hover,
.import-btn:focus {
  background-color: #184f9f;
  border-color: #184f9f;
}
</style>
