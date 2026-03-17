#!/bin/bash

# Aptamer微调训练启动脚本（单卡版本）
# 任务: 从预训练checkpoint进行全参微调
# 特点:
#   1. RNA类型在配置文件中指定（rna_type_token: sRNA）
#   2. 可选择是否使用物种层次分类学前缀
#   3. EOS token增加loss权重，帮助模型学会正确断句
#   4. 从预训练checkpoint继续训练（只加载模型权重，optimizer/scheduler重新初始化）
#
# 使用nohup在后台运行，单卡微调模式

set -e

# =============================================================================
# 默认配置参数（单卡版本）
# =============================================================================

# GPU配置（单卡模式）
DEFAULT_GPU_ID=0  # 默认使用GPU 0

# 容器配置
DEFAULT_CONTAINER_NAME="eva"

# 实验配置
EXPERIMENT_NAME=""
CONFIG_FILE=""
USER_CONFIG_FILE=""

# =============================================================================
# 【在这里设置默认配置文件路径】
# 如果设置了此路径，将优先使用，无需命令行指定
# 支持宿主机路径或容器路径
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/experiment_config_template.yaml"
# DEFAULT_CONFIG_FILE=""  # 留空则需要通过命令行指定

# 宿主机项目根目录
HOST_PROJECT_DIR="${PROJECT_ROOT}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "${BLUE}[====]${NC} $1"
}

# 参数初始化
GPU_ID=${DEFAULT_GPU_ID}
CONTAINER_NAME=${DEFAULT_CONTAINER_NAME}

# 参数解析和显示帮助
usage() {
    cat << EOF
Aptamer微调训练启动脚本（单卡版本）

使用方法：
    $0 [选项]
    $0 --config <配置文件路径> [选项]
    $0 <实验目录路径> [选项]

参数：
    实验目录路径                  直接传入实验目录路径
                                支持多种格式：
                                  1. 绝对路径: /path/to/results/aptamer_finetuning/xxx
                                  2. 相对路径: results/aptamer_finetuning/xxx
                                  3. 实验名称: aptamer_finetuning_v1
                                脚本将自动在实验目录下查找 experiment_config.yaml

可选选项：
    --config CONFIG_FILE        直接指定配置文件路径（优先级高于实验目录）
                                支持宿主机路径或容器路径
    --gpu GPU_ID                指定使用的GPU编号 (默认: ${DEFAULT_GPU_ID})
    --container CONTAINER       容器名称 (默认: ${DEFAULT_CONTAINER_NAME})
    --help                      显示此帮助信息

示例：
    # 使用默认配置文件（脚本内设置）
    $0

    # 直接指定配置文件
    $0 --config /path/to/my_config.yaml

    # 使用实验目录
    $0 aptamer_finetuning_v1

    # 指定GPU
    $0 --gpu 1 --config /path/to/config.yaml

说明：
    Aptamer微调是在预训练基础上的全参数微调，主要特点：
    1. RNA类型在配置文件中指定（rna_type_token: sRNA）
    2. 可选择是否使用物种层次分类学前缀
    3. EOS token增加loss权重，帮助模型学会正确断句
    4. 从预训练checkpoint继续训练（只加载模型权重，optimizer/scheduler重新初始化）

    配置文件中需要指定：
    - rna_type_token: RNA类型（如 sRNA, viral_RNA 等）
    - use_lineage_prefix: 是否使用物种前缀（true/false）
    - resume_from_pretrain: 预训练模型路径

EOF
}

# 检查容器状态
check_container() {
    if ! docker ps | grep -q "${CONTAINER_NAME}"; then
        log_error "${CONTAINER_NAME}容器未运行，请先启动容器"
        exit 1
    fi
}

# 创建输出目录
create_output_dirs() {
    docker exec ${CONTAINER_NAME} mkdir -p ${PROJECT_ROOT}/results/aptamer_finetuning
    docker exec ${CONTAINER_NAME} mkdir -p ${PROJECT_ROOT}/results/logs/aptamer_finetuning
}

# 主函数
main() {
    # 参数解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                USER_CONFIG_FILE="$2"
                shift 2
                ;;
            --gpu)
                GPU_ID="$2"
                shift 2
                ;;
            --container)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            -*)
                log_error "未知选项: $1"
                usage
                exit 1
                ;;
            *)
                # 第一个非选项参数作为实验目录路径
                if [ -z "$EXPERIMENT_NAME" ]; then
                    EXPERIMENT_NAME="$1"
                    shift
                else
                    log_error "多余的参数: $1"
                    usage
                    exit 1
                fi
                ;;
        esac
    done

    log_section "🚀 启动 Aptamer微调训练（单卡版本）"
    log_info "任务: 从预训练checkpoint进行全参微调"
    log_info "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 检查必需参数
    # 如果命令行没有指定配置文件，使用脚本内设置的默认路径
    if [ -z "$USER_CONFIG_FILE" ] && [ -n "$DEFAULT_CONFIG_FILE" ]; then
        USER_CONFIG_FILE="$DEFAULT_CONFIG_FILE"
        log_info "使用脚本内设置的默认配置文件"
    fi

    if [ -z "$USER_CONFIG_FILE" ] && [ -z "$EXPERIMENT_NAME" ]; then
        log_error "请指定实验目录路径或配置文件"
        log_info "用法: $0 --config <配置文件路径> [选项]"
        log_info "或者在脚本内设置 DEFAULT_CONFIG_FILE 变量"
        exit 1
    fi

    # 显示单卡配置
    log_info "单卡训练配置："
    log_info "  GPU编号: ${GPU_ID}"
    log_info "  容器名称: ${CONTAINER_NAME}"
    echo ""

    # 检查环境
    check_container

    log_info "查找实验配置..."

    # 智能处理路径：支持多种格式
    local container_path=""

    # 如果用户直接指定了配置文件，优先使用
    if [ -n "$USER_CONFIG_FILE" ]; then
        log_info "使用用户指定的配置文件"
        if [[ "$USER_CONFIG_FILE" == /* ]]; then
            CONFIG_FILE="$USER_CONFIG_FILE"
            log_info "检测到绝对路径"
        else
            # 相对路径，假设在项目根目录下
            CONFIG_FILE="${PROJECT_ROOT}/$USER_CONFIG_FILE"
            log_info "检测到相对路径"
        fi
    elif [[ "$EXPERIMENT_NAME" == /* ]]; then
        # 绝对路径，直接使用
        container_path="$EXPERIMENT_NAME"
        log_info "检测到绝对路径"
    elif [[ "$EXPERIMENT_NAME" == results/* ]]; then
        # 相对路径格式
        container_path="${PROJECT_ROOT}/$EXPERIMENT_NAME"
        log_info "检测到相对路径格式"
    else
        # 仅实验名称，构建完整路径
        container_path="${PROJECT_ROOT}/results/aptamer_finetuning/$EXPERIMENT_NAME"
        log_info "检测到实验名称，构建完整路径"
    fi

    # 如果不是用户直接指定配置文件，则从实验目录构建配置文件路径
    if [ -z "$USER_CONFIG_FILE" ]; then
        # 去除路径末尾的斜杠（如果有）
        container_path="${container_path%/}"

        # 构建配置文件路径
        CONFIG_FILE="${container_path}/experiment_config.yaml"
    fi

    # 检查配置文件是否存在
    if ! docker exec ${CONTAINER_NAME} test -f "$CONFIG_FILE"; then
        log_error "配置文件不存在: $CONFIG_FILE"
        log_info "请确保实验目录下有 experiment_config.yaml"
        exit 1
    fi

    # 创建输出目录
    create_output_dirs

    # 训练命令（单卡模式，使用python直接运行）
    TRAIN_CMD="cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${GPU_ID} python \
        finetune/train_finetune.py \
        --config=$CONFIG_FILE"

    # 日志文件
    LOG_DIR="${PROJECT_ROOT}/results/logs/aptamer_finetuning"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/aptamer_finetuning_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    log_info "📋 训练配置:"
    log_info "  - 配置文件: $CONFIG_FILE"
    log_info "  - 日志文件: $LOG_FILE"
    log_info "  - GPU编号: ${GPU_ID}"
    log_info "  - 任务类型: Aptamer微调（单卡全参数微调）"
    log_info "  - 预训练权重: 从checkpoint加载（optimizer/scheduler重新初始化）"
    echo ""

    log_info "📊 训练特性:"
    log_info "  ✓ RNA类型在配置文件中指定"
    log_info "  ✓ 可选物种层次分类学前缀"
    log_info "  ✓ EOS token loss权重增加"
    log_info "  ✓ 单卡微调模式"
    echo ""

    # 启动训练
    log_section "🎯 启动训练（单卡）..."

    # 使用nohup在后台运行（单卡模式，不需要NCCL配置）
    nohup docker exec \
        -e LD_LIBRARY_PATH=/usr/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
        ${CONTAINER_NAME} bash -c "$TRAIN_CMD" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!

    echo ""
    log_info "✅ 训练已启动（单卡）"
    log_info "  - 进程ID: $TRAIN_PID"
    log_info "  - 日志文件: $LOG_FILE"
    echo ""
    log_info "📊 监控命令:"
    log_info "  - 查看日志: tail -f $LOG_FILE"
    log_info "  - 检查进程: docker exec ${CONTAINER_NAME} ps aux | grep train_finetune"
    log_info "  - 停止训练: ./kill_training.sh"
    log_info "  - GPU状态: docker exec ${CONTAINER_NAME} nvidia-smi"
    echo ""
    log_info "📈 训练将输出到:"
    log_info "  - 模型检查点: ${PROJECT_ROOT}/results/aptamer_finetuning/"
    echo ""
}

# 执行主函数
main "$@"
