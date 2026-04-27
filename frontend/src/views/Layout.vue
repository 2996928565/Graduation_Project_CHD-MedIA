<template>
  <el-container class="layout-container">
    <!-- 侧边栏 -->
    <el-aside :width="collapsed ? '64px' : '220px'" class="aside">
      <div class="aside-header">
        <span v-if="!collapsed" class="brand-text">CHD-MedIA</span>
      </div>

      <el-menu
        :default-active="activeMenu"
        :collapse="collapsed"
        :router="true"
        background-color="#001529"
        text-color="#ffffffa0"
        active-text-color="#ffffff"
        class="side-menu"
      >
        <el-menu-item index="/patients">
          <el-icon><User /></el-icon>
          <template #title>患者管理</template>
        </el-menu-item>
        <el-menu-item index="/patients/new">
          <el-icon><UserFilled /></el-icon>
          <template #title>新增患者</template>
        </el-menu-item>
        <el-menu-item index="/ultrasound" v-if="false">
          <el-icon><VideoCamera /></el-icon>
          <template #title>超声检测</template>
        </el-menu-item>
        <el-menu-item index="/mri">
          <el-icon><PictureFilled /></el-icon>
          <template #title>影像检测</template>
        </el-menu-item>
        <el-menu-item index="/report">
          <el-icon><Document /></el-icon>
          <template #title>报告生成</template>
        </el-menu-item>

        <el-menu-item index="/admin/users" v-if="isAdmin">
          <el-icon><User /></el-icon>
          <template #title>用户管理</template>
        </el-menu-item>
      </el-menu>

      <div class="aside-footer">
        <el-button
          text
          :icon="collapsed ? 'Expand' : 'Fold'"
          style="color: #ffffff60"
          @click="collapsed = !collapsed"
        />
      </div>
    </el-aside>

    <el-container>
      <!-- 顶部导航 -->
      <el-header class="header">
        <div class="header-left">
          <span class="page-title">{{ currentTitle }}</span>
        </div>
        <div class="header-right">
          <el-tag type="success" effect="dark" size="small" style="margin-right:12px">
            系统运行中
          </el-tag>
          <el-dropdown @command="handleCommand">
            <el-button text>
              <el-icon><Avatar /></el-icon>
              {{ currentUserLabel }}
              <el-icon class="el-icon--right"><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="logout">
                  <el-icon><SwitchButton /></el-icon>退出登录
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>

      <!-- 主内容区 -->
      <el-main class="main-content">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/store/auth.js'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const collapsed = ref(false)
const activeMenu = computed(() => route.path)
const currentTitle = computed(() => route.meta?.title || 'CHD-MedIA')

const isAdmin = computed(() => (authStore.role || '').toLowerCase() === 'admin')
const currentUserLabel = computed(() => authStore.fullName || authStore.username || '账户')

function handleCommand(cmd) {
  if (cmd === 'logout') {
    authStore.logout()
    router.push('/login')
  }
}
</script>

<style scoped>
.layout-container {
  height: 100vh;
  overflow: hidden;
}

.aside {
  background-color: #001529;
  transition: width 0.2s;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.aside-header {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid #ffffff15;
}

.brand-text {
  color: #fff;
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 2px;
}

.side-menu {
  flex: 1;
  border-right: none !important;
}

.aside-footer {
  padding: 8px;
  border-top: 1px solid #ffffff15;
  text-align: center;
}

.header {
  background: #fff;
  border-bottom: 1px solid #e8eef5;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 56px;
}

.page-title {
  font-size: 16px;
  font-weight: 600;
  color: #1a3a5c;
}

.header-right {
  display: flex;
  align-items: center;
}

.main-content {
  background: #f0f5fb;
  padding: 24px;
  overflow-y: auto;
}
</style>
