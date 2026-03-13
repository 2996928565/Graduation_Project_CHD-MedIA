import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/store/auth.js'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { requiresAuth: false },
  },
  {
    path: '/',
    component: () => import('@/views/Layout.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        redirect: '/patients',
      },
      {
        path: 'patients',
        name: 'PatientList',
        component: () => import('@/views/PatientList.vue'),
        meta: { title: '患者管理' },
      },
      {
        path: 'patients/new',
        name: 'PatientNew',
        component: () => import('@/views/PatientForm.vue'),
        meta: { title: '新增患者' },
      },
      {
        path: 'patients/:id/edit',
        name: 'PatientEdit',
        component: () => import('@/views/PatientForm.vue'),
        meta: { title: '编辑患者' },
      },
      {
        path: 'ultrasound',
        name: 'UltrasoundDetection',
        component: () => import('@/views/UltrasoundDetection.vue'),
        meta: { title: '超声检测' },
      },
      {
        path: 'mri',
        name: 'MriDetection',
        component: () => import('@/views/MriDetection.vue'),
        meta: { title: 'MRI 检测' },
      },
      {
        path: 'report',
        name: 'ReportView',
        component: () => import('@/views/ReportView.vue'),
        meta: { title: '报告生成' },
      },
    ],
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/',
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 路由守卫：未登录跳转到登录页
router.beforeEach((to) => {
  const authStore = useAuthStore()
  if (to.meta.requiresAuth !== false && !authStore.isLoggedIn) {
    return { name: 'Login' }
  }
})

export default router
