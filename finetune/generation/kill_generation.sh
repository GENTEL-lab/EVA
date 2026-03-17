#!/bin/bash
# 停止所有RNA序列生成进程
# 用于终止 run_all_gpus.sh 启动的生成任务

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="eva"

echo "============================================"
echo "停止RNA序列生成进程"
echo "============================================"

# 检查容器是否运行
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "错误: 容器 ${CONTAINER} 未运行"
    exit 1
fi

# 查找并显示当前运行的生成进程
echo ""
echo "查找运行中的生成进程..."
PIDS=$(docker exec $CONTAINER pgrep -f "generate_rna.py" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "没有找到运行中的生成进程"
    exit 0
fi

echo "找到以下进程:"
docker exec $CONTAINER ps aux | head -1
for pid in $PIDS; do
    docker exec $CONTAINER ps aux | grep "^.*$pid " | grep -v grep
done

# 统计进程数
NUM_PROCS=$(echo "$PIDS" | wc -w)
echo ""
echo "共 ${NUM_PROCS} 个生成进程"

# 确认是否终止
echo ""
read -p "是否终止这些进程? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 终止进程
echo ""
echo "正在终止进程..."
for pid in $PIDS; do
    docker exec $CONTAINER kill $pid 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  已终止 PID: $pid"
    else
        echo "  终止失败 PID: $pid (可能已结束)"
    fi
done

# 等待进程结束
sleep 2

# 再次检查
REMAINING=$(docker exec $CONTAINER pgrep -f "generate_rna.py" 2>/dev/null | wc -w)
if [ "$REMAINING" -gt 0 ]; then
    echo ""
    echo "警告: 仍有 ${REMAINING} 个进程未终止，尝试强制终止..."
    docker exec $CONTAINER pkill -9 -f "generate_rna.py" 2>/dev/null
    sleep 1
fi

# 最终检查
FINAL=$(docker exec $CONTAINER pgrep -f "generate_rna.py" 2>/dev/null | wc -w)
echo ""
echo "============================================"
if [ "$FINAL" -eq 0 ]; then
    echo "所有生成进程已终止"
else
    echo "警告: 仍有 ${FINAL} 个进程运行"
fi
echo "============================================"

# 显示已生成的文件
echo ""
echo "已生成的fasta文件:"
ls -lh ${SCRIPT_DIR}/fasta/*.fasta 2>/dev/null || echo "  (无)"
