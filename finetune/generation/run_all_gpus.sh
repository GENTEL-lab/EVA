#!/bin/bash
# 并行运行GPU脚本，每个生成指定数量的序列
# 支持指定RNA类型和lineage信息
# 使用容器: eva
#
# 使用方法：
#   ./run_all_gpus.sh [选项]
#
# 示例：
#   ./run_all_gpus.sh --rna_type sRNA --num_seqs 1000              # Aptamer生成
#   ./run_all_gpus.sh --rna_type viral_RNA --lineage "D__Viruses;P__..." --checkpoint /path/to/ckpt  # Virus生成
#   ./run_all_gpus.sh --gpus "0,1,2,3" --num_seqs 5000             # 使用4张GPU
#   ./run_all_gpus.sh --temperature 0.8                            # 指定生成温度
#
# 按 Ctrl+C 可随时停止生成，脚本会自动合并已生成的fasta文件

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_SCRIPT_DIR="${PROJECT_ROOT}/finetune/generation"

# 默认参数
BATCH_SIZE=2
GPU_LIST="1,2,3"
TEMPERATURE=1
TOP_K=50
CHECKPOINT="./results/checkpoint"
INSTANCES_PER_GPU=1
NUM_SEQS_PER_GPU=600  # 每张GPU生成的序列数
RNA_TYPE="Y_RNA"            # RNA类型（如 Y_RNA, viral_RNA）
LINEAGE=""             # 谱系字符串
#D__Viruses;P__Pisuviricota;C__Pisoniviricetes;O__Picornavirales;F__Picornaviridae;G__Enterovirus;S__Enterovirus_B

# 全局变量用于跟踪状态
declare -a RUNNING_PIDS=()
declare -a GPU_ARRAY=()
INSTANCES_PER_GPU_GLOBAL=1
MERGED_FILE=""
TOTAL_SEQUENCES=0
IS_STOPPING=false

# 信号处理函数：捕获 Ctrl+C
cleanup() {
    if [ "$IS_STOPPING" = true ]; then
        return
    fi
    IS_STOPPING=true

    echo ""
    echo "=========================================="
    echo "检测到中断信号，正在停止生成..."
    echo "=========================================="

    # 停止所有运行中的生成进程
    echo "停止生成进程..."
    for pid in "${RUNNING_PIDS[@]}"; do
        if docker exec eva ps -p "$pid" >/dev/null 2>&1; then
            docker exec eva kill "$pid" 2>/dev/null
            echo "  已停止进程 PID: $pid"
        fi
    done

    # 等待进程结束
    sleep 2

    # 合并已生成的fasta文件
    echo ""
    echo "正在合并已生成的fasta文件..."
    merge_fasta_files

    # 删除分散的fasta文件
    echo ""
    echo "清理分散的fasta文件..."
    cleanup_fasta_files
    echo "清理完成!"

    echo ""
    echo "=========================================="
    echo "生成已停止，合并完成!"
    echo "合并文件: $MERGED_FILE"
    echo "已生成序列数: $(grep -c '^>' "$MERGED_FILE" 2>/dev/null || echo 0)"
    echo "=========================================="

    exit 0
}

# 合并fasta文件的函数
merge_fasta_files() {
    rm -f "$MERGED_FILE" 2>/dev/null

    # 按GPU顺序合并
    for gpu in "${GPU_ARRAY[@]}"; do
        for ((inst=0; inst<INSTANCES_PER_GPU_GLOBAL; inst++)); do
            if [ $INSTANCES_PER_GPU_GLOBAL -eq 1 ]; then
                PATTERN="${SCRIPT_DIR}/fasta/*_gpu${gpu}.fasta"
            else
                PATTERN="${SCRIPT_DIR}/fasta/*_gpu${gpu}_inst${inst}.fasta"
            fi

            for f in $PATTERN; do
                if [ -f "$f" ]; then
                    cat "$f" >> "$MERGED_FILE"
                    echo "  合并: $(basename $f)"
                fi
            done
        done
    done
}

# 清理分散的fasta文件函数
cleanup_fasta_files() {
    for gpu in "${GPU_ARRAY[@]}"; do
        for ((inst=0; inst<INSTANCES_PER_GPU_GLOBAL; inst++)); do
            if [ $INSTANCES_PER_GPU_GLOBAL -eq 1 ]; then
                PATTERN="${SCRIPT_DIR}/fasta/*_gpu${gpu}.fasta"
            else
                PATTERN="${SCRIPT_DIR}/fasta/*_gpu${gpu}_inst${inst}.fasta"
            fi

            for f in $PATTERN; do
                if [ -f "$f" ]; then
                    rm -f "$f"
                    echo "  删除: $(basename $f)"
                fi
            done
        done
    done
}

# 注册信号处理
trap cleanup SIGINT SIGTERM
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --instances_per_gpu)
            INSTANCES_PER_GPU="$2"
            shift 2
            ;;
        --num_seqs)
            NUM_SEQS_PER_GPU="$2"
            shift 2
            ;;
        --rna_type)
            RNA_TYPE="$2"
            shift 2
            ;;
        --lineage)
            LINEAGE="$2"
            shift 2
            ;;
        -h|--help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "参数说明:"
            echo "  --batch_size        批次大小（可选），默认2"
            echo "  --gpus              GPU列表（可选），用逗号分隔，默认0,1,2,3,4,5,6,7"
            echo "  --temperature       生成温度（可选），默认1"
            echo "  --top_k             top-k采样（可选），默认50"
            echo "  --checkpoint        模型权重路径（可选），容器内路径"
            echo "  --instances_per_gpu 每张GPU上部署的模型实例数（可选），默认1"
            echo "  --num_seqs          每张GPU生成的序列数（可选），默认1000"
            echo "  --rna_type          RNA类型，如 sRNA, viral_RNA 等"
            echo "  --lineage           谱系字符串，如 'D__Viruses;P__Pisuviricota;...'"
            echo ""
            echo "示例:"
            echo "  $0 --rna_type sRNA --num_seqs 1000                    # Aptamer生成"
            echo "  $0 --rna_type viral_RNA --lineage 'D__Viruses;...'    # Virus生成"
            echo "  $0 --gpus \"0,1,2,3\" --num_seqs 5000                   # 使用4张GPU"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 创建日志目录
mkdir -p "$LOG_DIR"

# 将GPU列表转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARRAY[@]}

# 计算每个实例应生成的序列数
NUM_SEQS_PER_INSTANCE=$((NUM_SEQS_PER_GPU / INSTANCES_PER_GPU))

TOTAL_INSTANCES=$((NUM_GPUS * INSTANCES_PER_GPU))
TOTAL_SEQUENCES=$((NUM_GPUS * NUM_SEQS_PER_GPU))

# 保存全局变量供信号处理函数使用
INSTANCES_PER_GPU_GLOBAL=$INSTANCES_PER_GPU

echo "=========================================="
echo "启动多GPU并行生成序列"
echo "容器: eva"
echo "使用GPU: ${GPU_LIST}"
echo "每张GPU实例数: ${INSTANCES_PER_GPU}"
echo "每张GPU序列数: ${NUM_SEQS_PER_GPU} (每实例: ${NUM_SEQS_PER_INSTANCE})"
echo "总计: ${TOTAL_SEQUENCES}条序列"
echo "总实例数: ${TOTAL_INSTANCES}"
echo "batchsize=${BATCH_SIZE}，temperature=${TEMPERATURE}，top_k=${TOP_K}"
echo "模型权重: ${CHECKPOINT}"
if [ -n "$RNA_TYPE" ]; then
    echo "RNA类型: ${RNA_TYPE}"
fi
if [ -n "$LINEAGE" ]; then
    echo "Lineage: ${LINEAGE}"
fi
echo "=========================================="

# 启动所有实例
for gpu in "${GPU_ARRAY[@]}"; do
    for ((inst=0; inst<INSTANCES_PER_GPU; inst++)); do
        if [ $INSTANCES_PER_GPU -eq 1 ]; then
            logfile="$LOG_DIR/gpu${gpu}.log"
            instance_name="GPU ${gpu}"
        else
            logfile="$LOG_DIR/gpu${gpu}_inst${inst}.log"
            instance_name="GPU ${gpu} 实例 ${inst}"
        fi

        echo "启动 ${instance_name}..."

        # 构建Python脚本的参数
        PYTHON_ARGS="--gpu ${gpu} --instance_id ${inst} --num_seqs ${NUM_SEQS_PER_INSTANCE} --batch_size ${BATCH_SIZE} --temperature ${TEMPERATURE} --top_k ${TOP_K} --checkpoint ${CHECKPOINT}"

        # 添加RNA类型参数
        if [ -n "$RNA_TYPE" ]; then
            PYTHON_ARGS="${PYTHON_ARGS} --rna_type ${RNA_TYPE}"
        fi

        # 添加lineage参数
        if [ -n "$LINEAGE" ]; then
            PYTHON_ARGS="${PYTHON_ARGS} --lineage '${LINEAGE}'"
        fi

        # 使用docker exec在容器中运行脚本，并获取进程PID
        CONTAINER_PID=$(docker exec eva bash -c "export CUDA_VISIBLE_DEVICES=${gpu} && nohup python ${CONTAINER_SCRIPT_DIR}/generate_rna.py ${PYTHON_ARGS} > ${CONTAINER_SCRIPT_DIR}/logs/gpu${gpu}$( [ $INSTANCES_PER_GPU -gt 1 ] && echo \"_inst${inst}\" ).log 2>&1 & echo \$!")
        RUNNING_PIDS+=("$CONTAINER_PID")

        echo "${instance_name}: 已启动 (PID: $CONTAINER_PID)"
    done
done

# 修改输出目录权限
docker exec eva bash -c "chmod -R 777 ${CONTAINER_SCRIPT_DIR}/fasta/ 2>/dev/null || true"
docker exec eva bash -c "chmod -R 777 ${CONTAINER_SCRIPT_DIR}/logs/ 2>/dev/null || true"

# 构建合并后的文件名
if [ -n "$RNA_TYPE" ] && [ -n "$LINEAGE" ]; then
    MERGED_FILE="${SCRIPT_DIR}/fasta/${RNA_TYPE}_lineage_${TOTAL_SEQUENCES}_merged.fasta"
elif [ -n "$RNA_TYPE" ]; then
    MERGED_FILE="${SCRIPT_DIR}/fasta/${RNA_TYPE}_${TOTAL_SEQUENCES}_merged.fasta"
elif [ -n "$LINEAGE" ]; then
    MERGED_FILE="${SCRIPT_DIR}/fasta/lineage_${TOTAL_SEQUENCES}_merged.fasta"
else
    MERGED_FILE="${SCRIPT_DIR}/fasta/unconditional_${TOTAL_SEQUENCES}_merged.fasta"
fi

echo ""
echo "所有实例已启动"
echo "日志文件位于: $LOG_DIR"
echo ""
echo "查看实时日志: tail -f $LOG_DIR/gpu<N>.log"
echo "查看所有进程: docker exec eva ps aux | grep generate_rna"
echo "查看GPU状态: docker exec eva nvidia-smi"
echo ""
echo "等待所有任务完成..."

# 等待所有生成进程完成
while true; do
    # 检查是否正在停止
    if [ "$IS_STOPPING" = true ]; then
        break
    fi

    # 检查是否还有generate_rna进程在运行
    RUNNING=$(docker exec eva pgrep -f "generate_rna.py" 2>/dev/null | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "所有生成任务已完成!"
        break
    fi

    # 显示当前进度
    CURRENT_COUNT=$(cat ${SCRIPT_DIR}/fasta/*.fasta 2>/dev/null | grep -c '^>' || echo 0)
    echo -ne "\r进度: ${CURRENT_COUNT}/${TOTAL_SEQUENCES} 条序列, ${RUNNING} 个进程运行中... (按 Ctrl+C 停止并合并)"
    sleep 5
done

# 如果不是被中断的，正常完成时合并文件
if [ "$IS_STOPPING" = false ]; then
    # 合并所有fasta文件
    echo ""
    echo "正在合并fasta文件..."
    merge_fasta_files

    # 统计最终结果
    FINAL_COUNT=$(grep -c '^>' "$MERGED_FILE" 2>/dev/null || echo 0)
    echo ""
    echo "=========================================="
    echo "生成完成!"
    echo "合并文件: $MERGED_FILE"
    echo "总序列数: $FINAL_COUNT"
    echo "=========================================="

    # 删除分散的fasta文件（保留merged文件）
    echo ""
    echo "清理分散的fasta文件..."
    cleanup_fasta_files
    echo "清理完成!"
fi