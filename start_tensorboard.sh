#!/usr/bin/env bash
# TensorBoard 启动脚本：正确 logdir + 可选外网可访问
# 用法: ./start_tensorboard.sh [logdir] [端口]
# 若 6006 被占用可先执行: fuser -k 6006/tcp  或改用端口 6007
LOG_DIR="${1:-/home/kyxmb/mkh/AMP/PB_AMP/logs/walk}"
PORT="${2:-6007}"
echo "TensorBoard logdir: $LOG_DIR"
echo "本地访问: http://localhost:$PORT/"
echo "远程时在本地终端执行: ssh -L ${PORT}:localhost:${PORT} 用户@本机 后再打开 http://localhost:$PORT/"
# --load_fast=false 可避免部分环境下界面无数据或报错
exec tensorboard --logdir="$LOG_DIR" --bind_all --port="$PORT" --load_fast=false
