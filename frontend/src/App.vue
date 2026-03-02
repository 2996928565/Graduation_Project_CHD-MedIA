<template>
  <router-view />
</template>

<style>
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
    'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
  background: #f0f5fb;
}

/* 全局滚动条样式 */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
::-webkit-scrollbar-track {
  background: #f1f5fb;
}
::-webkit-scrollbar-thumb {
  background: #b8cfe8;
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: #7da8d4;
}

/* Element Plus 全局调整 */
.el-card {
  border-radius: 10px !important;
}
.el-button {
  border-radius: 6px !important;
}
</style>
