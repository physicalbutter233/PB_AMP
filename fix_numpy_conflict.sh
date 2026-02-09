#!/bin/bash
# 修复 numpy 版本冲突问题
# 问题：~/.local/lib/python3.10/site-packages/ 中的 numpy 与 Isaac Sim 自带的 numpy 不兼容

echo "=== 修复 numpy 版本冲突 ==="
echo ""

# 检查是否存在冲突的 numpy
NUMPY_LOCAL_PATH="$HOME/.local/lib/python3.10/site-packages/numpy"
if [ -d "$NUMPY_LOCAL_PATH" ]; then
    echo "[INFO] 发现用户安装的 numpy: $NUMPY_LOCAL_PATH"
    echo "[INFO] 这将与 Isaac Sim 自带的 numpy 冲突"
    echo ""
    echo "解决方案："
    echo "1. 备份并移除用户安装的 numpy（推荐）"
    echo "2. 或者重命名它以避免冲突"
    echo ""
    read -p "是否要备份并移除用户安装的 numpy? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        BACKUP_DIR="$HOME/.local/lib/python3.10/site-packages/numpy_backup_$(date +%Y%m%d_%H%M%S)"
        echo "[INFO] 备份到: $BACKUP_DIR"
        mv "$NUMPY_LOCAL_PATH" "$BACKUP_DIR"
        mv "$HOME/.local/lib/python3.10/site-packages/numpy-2.2.6.dist-info" "${BACKUP_DIR}.dist-info" 2>/dev/null || true
        mv "$HOME/.local/lib/python3.10/site-packages/numpy.libs" "${BACKUP_DIR}.libs" 2>/dev/null || true
        echo "[SUCCESS] numpy 已备份并移除"
        echo "[INFO] 如果需要恢复，可以运行: mv $BACKUP_DIR $NUMPY_LOCAL_PATH"
    else
        echo "[INFO] 跳过移除操作"
        echo "[WARNING] 如果仍有问题，请手动移除或重命名: $NUMPY_LOCAL_PATH"
    fi
else
    echo "[INFO] 未发现用户安装的 numpy，无需修复"
fi

echo ""
echo "=== 检查 conda 环境中的 numpy ==="
if command -v conda &> /dev/null; then
    echo "[INFO] 当前 conda 环境: $CONDA_DEFAULT_ENV"
    echo "[INFO] 请确保在正确的 conda 环境中运行脚本"
    echo "[INFO] 如果问题仍然存在，请尝试："
    echo "  conda install numpy -y"
    echo "  或"
    echo "  pip uninstall numpy -y && conda install numpy -y"
fi

echo ""
echo "=== 完成 ==="
