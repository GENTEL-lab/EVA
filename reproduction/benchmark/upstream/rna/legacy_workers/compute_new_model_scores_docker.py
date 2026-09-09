#!/usr/bin/env python3
"""
新模型Docker Wrapper - 从宿主机调用docker容器计算log-likelihood（支持5'/3' tokens）

此脚本运行在宿主机（RhoDesign2_copy conda环境），通过docker exec
调用容器内的计算脚本，自动为序列添加5'和3'方向标记

与原有compute_rnagen_scores_docker.py的区别：
1. 调用新的Docker内脚本（compute_ll_for_new_models.py）
2. 支持任意checkpoint路径（不再硬编码模型路径）
3. 自动添加5'/3' tokens
4. 默认使用 `rnagen5` 容器（可通过 `RNAGEN_DOCKER_NAME` 环境变量覆盖）

用法：
    python compute_new_model_scores_docker.py <fasta_file> <checkpoint_path> <device>

参数：
    fasta_file: 宿主机上的FASTA文件路径
    checkpoint_path: 宿主机上的checkpoint路径
    device: 计算设备（cuda:0, cuda:1, cpu等）

返回：
    返回log-likelihood列表（JSON格式）
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path
import os


# Docker配置
DEFAULT_DOCKER_CONTAINER = "rnagen5"
DOCKER_CONTAINER_NAME = os.environ.get("RNAGEN_DOCKER_NAME", DEFAULT_DOCKER_CONTAINER)
DOCKER_TEMP_DIR = "/rna-multiverse/benchmark_rnagen/tmp"  # 容器内路径
HOST_TEMP_DIR = "/data4/huangyanjie/rna_benchmark/benchmark_rnagen/tmp"  # 宿主机路径


def check_docker_container():
    """检查docker容器是否运行"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=True
        )
        if DOCKER_CONTAINER_NAME in result.stdout:
            return True
        else:
            print(f"错误: Docker容器 '{DOCKER_CONTAINER_NAME}' 未运行", file=sys.stderr)
            print(f"请先启动容器，或查看可用容器: docker ps", file=sys.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"错误: 无法检查docker状态: {e}", file=sys.stderr)
        return False


def prepare_temp_directory():
    """准备临时目录"""
    os.makedirs(HOST_TEMP_DIR, exist_ok=True)


def copy_script_to_docker():
    """复制计算脚本到docker可访问位置"""
    script_dir = Path(__file__).parent
    local_script = script_dir / "compute_ll_for_new_models.py"

    if not local_script.exists():
        raise FileNotFoundError(f"计算脚本不存在: {local_script}")

    # 确保临时目录存在
    os.makedirs(HOST_TEMP_DIR, exist_ok=True)

    # 复制脚本到temp_benchmark目录
    target_path = os.path.join(HOST_TEMP_DIR, "compute_ll_for_new_models.py")
    shutil.copy(local_script, target_path)

    return target_path


def compute_log_likelihoods_via_docker(fasta_file, checkpoint_path, device='cuda:0', normalize=False):
    """
    通过docker调用计算log-likelihood

    Args:
        fasta_file: 宿主机FASTA文件路径
        checkpoint_path: 宿主机checkpoint路径
        device: 计算设备
        normalize: 是否归一化为PTLL（除以token数量）

    Returns:
        log_likelihoods: 对数似然值列表

    Raises:
        RuntimeError: 计算失败时抛出异常
    """
    # 1. 检查docker容器
    if not check_docker_container():
        raise RuntimeError(f"Docker容器 '{DOCKER_CONTAINER_NAME}' 未运行")

    # 2. 检查checkpoint路径是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint路径不存在: {checkpoint_path}")

    # 3. 准备临时目录
    prepare_temp_directory()

    # 4. 复制计算脚本到docker
    copy_script_to_docker()

    # 5. 复制FASTA文件到docker可访问位置
    fasta_filename = Path(fasta_file).name
    temp_fasta_host = os.path.join(HOST_TEMP_DIR, fasta_filename)
    temp_fasta_docker = f"{DOCKER_TEMP_DIR}/{fasta_filename}"

    shutil.copy(fasta_file, temp_fasta_host)

    # Docker内的脚本路径和checkpoint路径（使用绝对路径）
    docker_script_path = f"{DOCKER_TEMP_DIR}/compute_ll_for_new_models.py"

    # 转换checkpoint路径：宿主机路径 -> 容器内路径
    # 宿主机: /data4/huangyanjie/rna_benchmark/... -> 容器内: /rna-multiverse/...
    if checkpoint_path.startswith('/data4/huangyanjie/rna_benchmark/'):
        docker_checkpoint_path = checkpoint_path.replace('/data4/huangyanjie/rna_benchmark/', '/rna-multiverse/')
    else:
        docker_checkpoint_path = checkpoint_path  # 假设已经是容器内路径

    try:
        # 6. 调用docker exec运行计算
        # 注意：cd到rnagen项目目录，避免torch C扩展加载问题
        cmd = [
            'docker', 'exec', DOCKER_CONTAINER_NAME,
            'bash', '-c',
            f'cd /rna-multiverse/rnagen && '
            f'python3 {docker_script_path} {temp_fasta_docker} {docker_checkpoint_path} {device} {str(normalize).lower()}'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        # 7. 解析JSON结果（只取最后一行，过滤警告信息）
        try:
            # stdout可能包含警告信息，只解析最后一行JSON
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]  # 最后一行是JSON结果
            output = json.loads(json_line)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Docker输出:\n{result.stdout}", file=sys.stderr)
            print(f"Docker错误:\n{result.stderr}", file=sys.stderr)
            raise RuntimeError(f"无法解析docker返回的JSON: {e}")

        # 8. 检查是否成功
        if not output.get('success', False):
            error_msg = output.get('error', 'Unknown error')
            print(f"DEBUG: Docker returned failure. Full output: {output}", file=sys.stderr)
            raise RuntimeError(f"Docker计算失败: {error_msg}")

        return output['log_likelihoods']

    finally:
        # 9. 清理临时文件
        if os.path.exists(temp_fasta_host):
            os.remove(temp_fasta_host)


def main():
    """命令行接口"""
    if len(sys.argv) < 3:
        print("用法: python compute_new_model_scores_docker.py <fasta_file> <checkpoint_path> [device] [normalize]")
        print("示例: python compute_new_model_scores_docker.py data.fasta /path/to/checkpoint cuda:0 false")
        sys.exit(1)

    fasta_file = sys.argv[1]
    checkpoint_path = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else 'cuda:0'
    # 可选参数: normalize (默认为False)
    normalize = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False

    if not os.path.exists(fasta_file):
        print(f"错误: FASTA文件不存在: {fasta_file}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"错误: Checkpoint路径不存在: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    try:
        log_likelihoods = compute_log_likelihoods_via_docker(fasta_file, checkpoint_path, device, normalize=normalize)

        # 输出结果（JSON格式）
        result = {
            'log_likelihoods': log_likelihoods,
            'num_sequences': len(log_likelihoods)
        }
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
