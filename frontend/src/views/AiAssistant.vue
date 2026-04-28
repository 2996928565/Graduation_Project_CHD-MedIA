<template>
  <div class="assistant-page">
    <div class="page-header">
      <h2><el-icon><ChatDotRound /></el-icon> 智能问答助手</h2>
      <el-tag type="success">千问模型</el-tag>
    </div>

    <el-row :gutter="16">
      <el-col :span="8">
        <el-card shadow="never" class="left-card">
          <template #header>
            <span class="card-title">患者检测结果导入</span>
          </template>

          <el-form label-position="top" @submit.prevent>
            <el-form-item label="选择患者">
              <el-select
                v-model="selectedPatientId"
                filterable
                placeholder="请选择患者"
                style="width: 100%"
                :loading="patientsLoading"
              >
                <el-option
                  v-for="p in patients"
                  :key="p.patient_id"
                  :label="`${p.name}（${p.sex || '未知'} ${p.age ?? '-'}岁）`"
                  :value="p.patient_id"
                />
              </el-select>
            </el-form-item>

            <el-form-item label="导入最近记录数">
              <el-input-number v-model="importLimit" :min="1" :max="30" style="width: 100%" />
            </el-form-item>

            <el-form-item>
              <el-button
                type="primary"
                style="width: 100%"
                :loading="importing"
                :disabled="!selectedPatientId"
                @click="handleImportContext"
              >
                导入该患者检测结果
              </el-button>
            </el-form-item>
          </el-form>

          <el-divider content-position="left">已导入记录</el-divider>

          <el-empty
            v-if="!importedRecords.length"
            description="请先选择患者并导入检测结果"
          />

          <el-scrollbar v-else height="340px">
            <div
              v-for="item in importedRecords"
              :key="item.task_id"
              class="record-item"
              :class="{ selected: selectedTaskIds.includes(item.task_id) }"
              @click="toggleTask(item.task_id)"
            >
              <div class="record-title">
                <el-checkbox
                  :model-value="selectedTaskIds.includes(item.task_id)"
                  @change="() => toggleTask(item.task_id)"
                  @click.stop
                />
                <span class="task-id">{{ item.task_id }}</span>
              </div>
              <div class="record-meta">
                <el-tag :type="item.modality === 'mri' ? 'warning' : 'primary'" size="small">
                  {{ item.modality === 'mri' ? '影像' : '超声' }}
                </el-tag>
                <span>{{ item.detections_count }} 项检测</span>
              </div>
              <div class="record-sub">{{ item.filename }}</div>
              <div class="record-sub">{{ formatDate(item.created_at) }}</div>
            </div>
          </el-scrollbar>

          <div v-if="importedRecords.length" class="record-actions">
            <el-button text @click="selectAllTasks">全选</el-button>
            <el-button text @click="clearTaskSelection">清空</el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card shadow="never" class="chat-card">
          <template #header>
            <div class="chat-header">
              <span class="card-title">问答对话</span>
              <span class="context-tip">
                {{ selectedTaskIds.length ? `已挂载 ${selectedTaskIds.length} 条检测记录` : '未挂载检测记录' }}
              </span>
            </div>
          </template>

          <el-scrollbar ref="chatScrollRef" height="430px" class="chat-scroll">
            <div v-if="!messages.length" class="empty-chat">
              <el-empty description="导入患者检测结果后开始提问，例如：是否提示室间隔缺损？" />
            </div>

            <div v-for="(msg, idx) in messages" :key="idx" class="msg-row" :class="msg.role">
              <div class="msg-bubble">
                <div class="msg-role">{{ msg.role === 'user' ? '我' : '助手' }}</div>
                <div class="msg-content">{{ msg.content }}</div>
                <div v-if="msg.meta" class="msg-meta">{{ msg.meta }}</div>
              </div>
            </div>
          </el-scrollbar>

          <div class="composer">
            <el-input
              v-model="question"
              type="textarea"
              :rows="3"
              maxlength="2000"
              show-word-limit
              placeholder="请输入你的问题，例如：根据当前导入结果，最需要关注哪些异常？"
              @keydown.ctrl.enter.prevent="handleAsk"
            />
            <div class="composer-actions">
              <el-button @click="clearMessages">清空对话</el-button>
              <el-button type="primary" :loading="asking" @click="handleAsk">
                发送（Ctrl+Enter）
              </el-button>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { nextTick, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { getPatients } from '@/api/patients.js'
import { getPatientDetectionContext, chatWithAssistant } from '@/api/assistant.js'

const patients = ref([])
const patientsLoading = ref(false)
const importing = ref(false)
const asking = ref(false)

const selectedPatientId = ref('')
const importLimit = ref(10)
const importedRecords = ref([])
const selectedTaskIds = ref([])

const question = ref('')
const messages = ref([])
const chatScrollRef = ref(null)

onMounted(fetchPatients)

async function fetchPatients() {
  patientsLoading.value = true
  try {
    const list = await getPatients()
    patients.value = Array.isArray(list) ? list : []
  } finally {
    patientsLoading.value = false
  }
}

async function handleImportContext() {
  if (!selectedPatientId.value) {
    ElMessage.warning('请先选择患者')
    return
  }

  importing.value = true
  try {
    const res = await getPatientDetectionContext(selectedPatientId.value, importLimit.value)
    importedRecords.value = res.records || []
    selectedTaskIds.value = importedRecords.value.map((r) => r.task_id)

    if (!importedRecords.value.length) {
      ElMessage.warning('该患者暂无检测记录')
      return
    }

    ElMessage.success(`已导入 ${importedRecords.value.length} 条检测记录`)
  } finally {
    importing.value = false
  }
}

function toggleTask(taskId) {
  const idx = selectedTaskIds.value.indexOf(taskId)
  if (idx >= 0) {
    selectedTaskIds.value.splice(idx, 1)
  } else {
    selectedTaskIds.value.push(taskId)
  }
}

function selectAllTasks() {
  selectedTaskIds.value = importedRecords.value.map((r) => r.task_id)
}

function clearTaskSelection() {
  selectedTaskIds.value = []
}

async function handleAsk() {
  const q = question.value.trim()
  if (!q) {
    ElMessage.warning('请输入问题')
    return
  }

  messages.value.push({ role: 'user', content: q })
  question.value = ''
  await scrollToBottom()

  asking.value = true
  try {
    const res = await chatWithAssistant({
      question: q,
      patient_id: selectedPatientId.value || null,
      task_ids: selectedTaskIds.value,
      history: messages.value.slice(-10).map((m) => ({ role: m.role, content: m.content })),
    })

    const source = res.source === 'qwen_llm' ? '千问模型' : '规则回退'
    const meta = `来源：${source}｜上下文记录：${res.context_records || 0}`

    messages.value.push({
      role: 'assistant',
      content: res.answer || '未返回回答内容',
      meta,
    })

    await scrollToBottom()
  } finally {
    asking.value = false
  }
}

function clearMessages() {
  messages.value = []
}

function formatDate(iso) {
  return iso ? iso.replace('T', ' ').slice(0, 19) : ''
}

async function scrollToBottom() {
  await nextTick()
  const wrap = chatScrollRef.value?.wrapRef
  if (wrap) {
    wrap.scrollTop = wrap.scrollHeight
  }
}
</script>

<style scoped>
.assistant-page {
  min-height: calc(100vh - 140px);
}

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

.left-card {
  height: 100%;
}

.record-item {
  border: 1px solid #e6edf5;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.record-item:hover {
  border-color: #7da8d6;
  background: #f7fbff;
}

.record-item.selected {
  border-color: #3a7bd5;
  background: #eef6ff;
}

.record-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.task-id {
  color: #2b4d72;
  font-size: 13px;
  font-weight: 600;
}

.record-meta {
  margin-top: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #5c7a99;
  font-size: 12px;
}

.record-sub {
  margin-top: 4px;
  color: #7d92a8;
  font-size: 12px;
  word-break: break-all;
}

.record-actions {
  margin-top: 6px;
  text-align: right;
}

.chat-card {
  height: 100%;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.context-tip {
  font-size: 12px;
  color: #6b87a6;
}

.chat-scroll {
  border: 1px solid #e8eef5;
  border-radius: 8px;
  background: #f9fbfe;
  padding: 8px;
}

.empty-chat {
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.msg-row {
  display: flex;
  margin: 10px 0;
}

.msg-row.user {
  justify-content: flex-end;
}

.msg-row.assistant {
  justify-content: flex-start;
}

.msg-bubble {
  max-width: 82%;
  border-radius: 10px;
  padding: 10px 12px;
  line-height: 1.6;
}

.msg-row.user .msg-bubble {
  background: #3a7bd5;
  color: #fff;
}

.msg-row.assistant .msg-bubble {
  background: #fff;
  border: 1px solid #dfe8f2;
  color: #22384f;
}

.msg-role {
  font-size: 12px;
  opacity: 0.8;
  margin-bottom: 4px;
}

.msg-content {
  white-space: pre-wrap;
  word-break: break-word;
}

.msg-meta {
  margin-top: 6px;
  font-size: 12px;
  color: #6d8198;
}

.composer {
  margin-top: 12px;
}

.composer-actions {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
</style>
