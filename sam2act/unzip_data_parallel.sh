#!/bin/bash
# 并行解压数据集的脚本
# 使用方法: bash unzip_data_parallel.sh

set -e

DATA_SOURCE="/home/hongzefu/sam2act/sam2act/data/data"
DATA_TARGET="/home/hongzefu/sam2act/sam2act/data"

# 创建目标目录
mkdir -p "$DATA_TARGET/train" "$DATA_TARGET/val" "$DATA_TARGET/test"

# 解压函数
unzip_task() {
    local split=$1
    local zip_file=$2
    local target_dir="$DATA_TARGET/$split"
    local zip_name=$(basename "$zip_file" .zip)
    
    echo "[$(date +'%H:%M:%S')] 开始解压 $split/$zip_name..."
    if unzip -q "$zip_file" -d "$target_dir" 2>/dev/null; then
        echo "[$(date +'%H:%M:%S')] ✓ 完成解压 $split/$zip_name"
    else
        echo "[$(date +'%H:%M:%S')] ✗ 解压失败 $split/$zip_name"
        return 1
    fi
}

# 导出函数以便并行使用
export -f unzip_task
export DATA_SOURCE DATA_TARGET

# 并行解压 train 数据
echo "=========================================="
echo "开始并行解压 train 数据 (使用 8 个并行进程)..."
echo "=========================================="
cd "$DATA_SOURCE/train"
find . -maxdepth 1 -name "*.zip" -type f | while read zip_file; do
    echo "$zip_file"
done | xargs -n 1 -P 8 -I {} bash -c 'unzip_task "train" "$DATA_SOURCE/train/{}"'

# 并行解压 val 数据
echo ""
echo "=========================================="
echo "开始并行解压 val 数据 (使用 8 个并行进程)..."
echo "=========================================="
cd "$DATA_SOURCE/val"
find . -maxdepth 1 -name "*.zip" -type f | while read zip_file; do
    echo "$zip_file"
done | xargs -n 1 -P 8 -I {} bash -c 'unzip_task "val" "$DATA_SOURCE/val/{}"'

# 并行解压 test 数据
echo ""
echo "=========================================="
echo "开始并行解压 test 数据 (使用 8 个并行进程)..."
echo "=========================================="
cd "$DATA_SOURCE/test"
find . -maxdepth 1 -name "*.zip" -type f | while read zip_file; do
    echo "$zip_file"
done | xargs -n 1 -P 8 -I {} bash -c 'unzip_task "test" "$DATA_SOURCE/test/{}"'

echo ""
echo "=========================================="
echo "所有数据解压完成！"
echo "=========================================="
echo "数据位置:"
echo "  Train: $DATA_TARGET/train"
echo "  Val:   $DATA_TARGET/val"
echo "  Test:  $DATA_TARGET/test"
echo ""
echo "验证解压结果:"
ls -d "$DATA_TARGET/train"/*/ 2>/dev/null | wc -l | xargs echo "  Train 任务数:"
ls -d "$DATA_TARGET/val"/*/ 2>/dev/null | wc -l | xargs echo "  Val 任务数:"
ls -d "$DATA_TARGET/test"/*/ 2>/dev/null | wc -l | xargs echo "  Test 任务数:"

