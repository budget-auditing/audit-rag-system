<script setup lang="ts">
import { useRoute } from 'vue-router'
const route = useRoute()
</script>

<template>
  <el-container style="height: 100vh">
    <el-aside width="200px" style="background: #1e293b; color: #fff">
      <div style="padding: 1rem; font-weight: bold; font-size: 1.1rem">审计 RAG 平台</div>
      <el-menu
        :default-active="route.path"
        background-color="#1e293b"
        text-color="#fff"
        active-text-color="#67c23a"
        router
      >
        <el-menu-item index="/dashboard">
          <span>系统概览</span>
        </el-menu-item>
        <el-menu-item index="/documents">
          <span>文档管理</span>
        </el-menu-item>
        <el-menu-item index="/index">
          <span>索引构建</span>
        </el-menu-item>
        <el-menu-item index="/qa">
          <span>审计问答</span>
        </el-menu-item>
        <el-menu-item index="/logs">
          <span>日志与导出</span>
        </el-menu-item>
        <el-menu-item index="/status">
          <span>系统状态</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-main>
      <router-view />
    </el-main>
  </el-container>
</template>

<style>
html, body { margin: 0; padding: 0; }
</style>
