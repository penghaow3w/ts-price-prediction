#!/bin/bash
# ========================================
# GitHub 仓库创建 & SSH 推送脚本
# GitHub Repository Creation & SSH Push Script
# ========================================
#
# 使用方法 / Usage:
#   1. 确保已配置SSH密钥到GitHub
#      Make sure SSH key is added to GitHub
#   2. 运行此脚本 / Run this script
#
# 注意：需要先在GitHub上手动创建仓库（如果API不可用）
# Note: Create repo on GitHub first if API is unavailable
# ========================================

REPO_NAME="ts-price-prediction"
GITHUB_USER="penghaow3w"
REPO_URL="git@github.com:${GITHUB_USER}/${REPO_NAME}.git"

echo "=========================================="
echo "  TS Price Prediction - GitHub Push"
echo "=========================================="

# 检查SSH连接
echo ""
echo "[1/4] 检查SSH连接 / Checking SSH connection..."
ssh -T git@github.com 2>&1 || true
echo ""

# 如果仓库不存在，使用GitHub API创建
echo "[2/4] 创建远程仓库 / Creating remote repository..."
# 请替换 YOUR_GITHUB_TOKEN 为你的Personal Access Token
# Please replace YOUR_GITHUB_TOKEN with your Personal Access Token
# TOKEN="YOUR_GITHUB_TOKEN"

# 方法1: 使用API创建（需要token）
# if [ -n "$TOKEN" ]; then
#     curl -X POST -H "Authorization: token $TOKEN" \
#         -H "Accept: application/vnd.github.v3+json" \
#         https://api.github.com/user/repos \
#         -d "{\"name\":\"$REPO_NAME\",\"description\":\"Stock & Crypto Price Prediction System | LSTM + Transformer + Prophet\",\"private\":false,\"auto_init\":false}"
#     echo ""
# fi

# 方法2: 提示用户手动创建
echo "如果仓库不存在，请先在GitHub上创建："
echo "If repo doesn't exist, create it at: https://github.com/new"
echo "  Repository name: $REPO_NAME"
echo "  Description: Stock & Crypto Price Prediction System | LSTM + Transformer + Prophet"
echo ""

# 添加远程仓库
echo "[3/4] 配置远程仓库 / Setting up remote..."
cd /workspace/ts-price-prediction
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
echo "  Remote: $REPO_URL"
echo ""

# 推送
echo "[4/4] 推送代码 / Pushing code..."
git push -u origin main

echo ""
echo "=========================================="
echo "  ✓ 完成 / Done!"
echo "  仓库地址 / Repo URL:"
echo "  https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo "=========================================="
