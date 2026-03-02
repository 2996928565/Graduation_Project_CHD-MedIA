<!--
  ImageUpload 组件
  支持拖拽上传、点击上传，自动调用预览接口（DICOM 解析）。
  Events:
    file-selected: { file, previewBase64, metadata }
-->
<template>
  <div class="upload-wrapper">
    <el-upload
      ref="uploadRef"
      drag
      :auto-upload="false"
      :accept="accept"
      :limit="1"
      :on-change="handleChange"
      :on-exceed="handleExceed"
      :show-file-list="false"
    >
      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        拖拽影像到此处，或 <em>点击上传</em>
      </div>
      <template #tip>
        <div class="upload-tip">
          支持格式：{{ accept }}<br />
          <span v-if="modality === 'mri'">MRI 序列：DICOM (.dcm) / PNG / JPG</span>
          <span v-else>超声影像：DICOM (.dcm) / PNG / JPG</span>
          <br />最大 {{ maxSizeMB }} MB
        </div>
      </template>
    </el-upload>

    <!-- 文件信息 -->
    <div v-if="selectedFileInfo" class="file-info">
      <el-icon><Document /></el-icon>
      <span>{{ selectedFileInfo.name }}</span>
      <el-tag size="small" style="margin-left:8px">
        {{ (selectedFileInfo.size / 1024).toFixed(1) }} KB
      </el-tag>
      <el-tag v-if="isDicom" type="warning" size="small" style="margin-left:4px">DICOM</el-tag>
      <el-button
        text
        type="danger"
        size="small"
        icon="Delete"
        style="margin-left:auto"
        @click="clearFile"
      />
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-tip">
      <el-icon class="is-loading"><Loading /></el-icon>
      {{ isDicom ? '正在解析 DICOM 文件...' : '正在加载影像...' }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { uploadPreview } from '@/api/images.js'

const props = defineProps({
  modality: { type: String, default: 'ultrasound' },
  accept: { type: String, default: '.png,.jpg,.jpeg,.dcm,.dicom' },
  maxSizeMB: { type: Number, default: 200 },
})

const emit = defineEmits(['file-selected'])

const uploadRef = ref(null)
const selectedFileInfo = ref(null)
const isDicom = ref(false)
const loading = ref(false)

async function handleChange(uploadFile) {
  const file = uploadFile.raw
  if (!file) return

  const sizeMB = file.size / (1024 * 1024)
  if (sizeMB > props.maxSizeMB) {
    ElMessage.error(`文件过大（${sizeMB.toFixed(1)} MB），最大支持 ${props.maxSizeMB} MB`)
    return
  }

  const ext = file.name.split('.').pop().toLowerCase()
  isDicom.value = ['dcm', 'dicom'].includes(ext)
  selectedFileInfo.value = { name: file.name, size: file.size }
  loading.value = true

  try {
    // 调用后端预览接口（DICOM 解析 + 转 PNG）
    const res = await uploadPreview(file)
    emit('file-selected', {
      file,
      previewBase64: res.preview_image_base64,
      metadata: res.metadata || null,
    })
  } catch {
    // 错误已在拦截器处理，降级为本地预览
    const reader = new FileReader()
    reader.onload = (e) => {
      emit('file-selected', { file, previewBase64: null, metadata: null })
    }
    reader.readAsArrayBuffer(file)
  } finally {
    loading.value = false
  }
}

function handleExceed() {
  ElMessage.warning('每次只能上传一个影像文件，请先删除当前文件')
}

function clearFile() {
  selectedFileInfo.value = null
  isDicom.value = false
  uploadRef.value?.clearFiles()
  emit('file-selected', { file: null, previewBase64: null, metadata: null })
}
</script>

<style scoped>
.upload-wrapper {
  width: 100%;
}
.upload-tip {
  font-size: 12px;
  color: #8ea8c3;
  margin-top: 6px;
  line-height: 1.8;
}
.file-info {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: #f0f7ff;
  border-radius: 6px;
  margin-top: 8px;
  font-size: 13px;
  color: #2c5f8a;
}
.loading-tip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  font-size: 13px;
  color: #5a7fa0;
}
</style>
