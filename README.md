# movie_recommendation
# 🎥 GCN-Based Movie Recommendation System with Clustering

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub stars](https://img.shields.io/github/stars/yourusername/gcn-recommendation-system?style=social)

一个基于图卷积网络(GCN)和K-means聚类的电影推荐系统，提供从数据预处理到模型评估的完整流程。

## 📌 目录

- [项目概述](#-项目概述)
- [✨ 功能特性](#-功能特性)
- [⚙️ 系统架构](#️-系统架构)
- [🚀 快速开始](#-快速开始)
  - [环境配置](#环境配置)
  - [数据准备](#数据准备)
  - [运行流程](#运行流程)
- [📂 项目结构](#-项目结构)
- [⚡ 核心模块](#-核心模块)
  - [GCN模型](#gcn模型)
  - [推荐生成](#推荐生成)
  - [评估系统](#评估系统)
- [📈 性能指标](#-性能指标)
- [🔧 配置说明](#-配置说明)
- [🤝 如何贡献](#-如何贡献)
- [📜 许可证](#-许可证)
- [🙏 致谢](#-致谢)

## 🌟 项目概述

本项目实现了一个端到端的电影推荐系统，核心技术栈：

- **图神经网络**：使用PyTorch Geometric实现的两层GCN网络
- **聚类算法**：K-means用户分组
- **推荐策略**：基于用户-物品嵌入相似度
- **评估体系**：多维度指标评估

完整工作流程：
```mermaid
graph TD
    A[原始数据] --> B[数据预处理]
    B --> C[GCN模型训练]
    C --> D[生成嵌入]
    D --> E[用户聚类]
    E --> F[推荐生成]
    F --> G[系统评估]
    ```
##✨ 功能特性
核心技术
- **基于消息传递的图卷积网络**

- **用户群体聚类分析**

- **Top-K推荐生成**

特色功能
✅ 完整的训练-推理-评估流水线
✅ 灵活的YAML配置系统
✅ 详细的日志和可视化输出
✅ 支持CPU/GPU加速
✅ 模块化设计易于扩展
    