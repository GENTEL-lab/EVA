#!/usr/bin/env python3
"""
RNA-ProGen3 结果分析脚本
提供深入的训练结果分析和可视化
"""

import os
import sys
import json
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression

# 设置项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加eval目录到路径
from scripts.path_utils import setup_project_paths
setup_project_paths()

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'analysis'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=getattr(logging, config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 设置matplotlib
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_experiment_data(self, data_path: str) -> pd.DataFrame:
        """加载实验数据"""
        data_path = Path(data_path)
        
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 假设是checkpoint_metrics.json格式
            if isinstance(data, list):
                rows = []
                for checkpoint in data:
                    metrics = checkpoint['metrics']
                    model_config = checkpoint.get('model_config', {})
                    
                    row = {
                        'checkpoint': checkpoint['checkpoint_name'],
                        'step': int(checkpoint['checkpoint_name'].split('-')[1]) if '-' in checkpoint['checkpoint_name'] else 0,
                        'val_loss': metrics.get('val_loss'),
                        'val_perplexity': metrics.get('val_perplexity'),
                        'total_tokens': metrics.get('total_tokens'),
                        'evaluation_time': checkpoint.get('evaluation_time'),
                        'model_size': model_config.get('hidden_size'),
                        'num_layers': model_config.get('num_hidden_layers'),
                        'num_experts': model_config.get('num_experts')
                    }
                    rows.append(row)
                
                df = pd.DataFrame(rows)
            else:
                raise ValueError("不支持的JSON格式")
        else:
            raise ValueError(f"不支持的文件格式: {data_path.suffix}")
        
        logger.info(f"加载数据: {len(df)} 条记录")
        return df
    
    def basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """基础统计分析"""
        stats_dict = {}
        
        # 数值列的描述统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_dict['descriptive'] = df[numeric_cols].describe()
        
        # 损失和困惑度统计
        if 'val_loss' in df.columns:
            stats_dict['loss_stats'] = {
                'min': df['val_loss'].min(),
                'max': df['val_loss'].max(),
                'mean': df['val_loss'].mean(),
                'std': df['val_loss'].std(),
                'improvement': df['val_loss'].iloc[0] - df['val_loss'].iloc[-1] if len(df) > 1 else 0
            }
        
        if 'val_perplexity' in df.columns:
            valid_ppl = df[df['val_perplexity'] < 1000]['val_perplexity']
            if not valid_ppl.empty:
                stats_dict['perplexity_stats'] = {
                    'min': valid_ppl.min(),
                    'max': valid_ppl.max(),
                    'mean': valid_ppl.mean(),
                    'std': valid_ppl.std()
                }
        
        # 训练效率统计
        if 'evaluation_time' in df.columns:
            stats_dict['efficiency_stats'] = {
                'avg_eval_time': df['evaluation_time'].mean(),
                'total_eval_time': df['evaluation_time'].sum()
            }
        
        return stats_dict
    
    def training_dynamics_analysis(self, df: pd.DataFrame):
        """训练动态分析"""
        if len(df) < 2:
            logger.warning("数据点不足，跳过训练动态分析")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 损失曲线
        if 'val_loss' in df.columns:
            axes[0, 0].plot(df['step'], df['val_loss'], marker='o', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('训练步数')
            axes[0, 0].set_ylabel('验证损失')
            axes[0, 0].set_title('验证损失变化')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(df) > 2:
                z = np.polyfit(df['step'], df['val_loss'], 1)
                p = np.poly1d(z)
                axes[0, 0].plot(df['step'], p(df['step']), "--", alpha=0.8, color='red', 
                              label=f'趋势: {z[0]:.6f}x + {z[1]:.3f}')
                axes[0, 0].legend()
        
        # 2. 困惑度曲线
        if 'val_perplexity' in df.columns:
            valid_ppl = df[df['val_perplexity'] < 1000]
            if not valid_ppl.empty:
                axes[0, 1].plot(valid_ppl['step'], valid_ppl['val_perplexity'], 
                              marker='o', linewidth=2, markersize=6, color='orange')
                axes[0, 1].set_xlabel('训练步数')
                axes[0, 1].set_ylabel('验证困惑度')
                axes[0, 1].set_title('验证困惑度变化')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 损失改善率
        if 'val_loss' in df.columns and len(df) > 1:
            loss_improvement = (df['val_loss'].iloc[0] - df['val_loss']) / df['val_loss'].iloc[0] * 100
            axes[1, 0].plot(df['step'], loss_improvement, marker='s', linewidth=2, 
                          markersize=6, color='green')
            axes[1, 0].set_xlabel('训练步数')
            axes[1, 0].set_ylabel('损失改善率 (%)')
            axes[1, 0].set_title('相对于初始损失的改善百分比')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 收敛速度分析
        if 'val_loss' in df.columns and len(df) > 2:
            # 计算损失变化率
            loss_diff = df['val_loss'].diff()
            step_diff = df['step'].diff()
            convergence_rate = -loss_diff / step_diff  # 负号表示损失下降的速度
            
            axes[1, 1].plot(df['step'][1:], convergence_rate[1:], marker='^', 
                          linewidth=2, markersize=6, color='purple')
            axes[1, 1].set_xlabel('训练步数')
            axes[1, 1].set_ylabel('收敛速度 (损失下降/步数)')
            axes[1, 1].set_title('收敛速度变化')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("训练动态分析图表已保存")
    
    def performance_distribution_analysis(self, df: pd.DataFrame):
        """性能分布分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 损失分布直方图
        if 'val_loss' in df.columns:
            axes[0, 0].hist(df['val_loss'], bins=min(10, len(df)//2), alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('验证损失')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].set_title('验证损失分布')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_loss = df['val_loss'].mean()
            std_loss = df['val_loss'].std()
            axes[0, 0].axvline(mean_loss, color='red', linestyle='--', 
                             label=f'均值: {mean_loss:.4f}')
            axes[0, 0].axvline(mean_loss + std_loss, color='orange', linestyle='--', alpha=0.7,
                             label=f'±1σ: {std_loss:.4f}')
            axes[0, 0].axvline(mean_loss - std_loss, color='orange', linestyle='--', alpha=0.7)
            axes[0, 0].legend()
        
        # 2. 困惑度分布（如果有效）
        if 'val_perplexity' in df.columns:
            valid_ppl = df[df['val_perplexity'] < 1000]['val_perplexity']
            if not valid_ppl.empty:
                axes[0, 1].hist(valid_ppl, bins=min(10, len(valid_ppl)//2), 
                              alpha=0.7, color='lightcoral', edgecolor='black')
                axes[0, 1].set_xlabel('验证困惑度')
                axes[0, 1].set_ylabel('频数')
                axes[0, 1].set_title('验证困惑度分布')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 损失vs步数散点图
        if 'val_loss' in df.columns and 'step' in df.columns:
            scatter = axes[1, 0].scatter(df['step'], df['val_loss'], 
                                       c=range(len(df)), cmap='viridis', s=60, alpha=0.7)
            axes[1, 0].set_xlabel('训练步数')
            axes[1, 0].set_ylabel('验证损失')
            axes[1, 0].set_title('损失vs训练步数')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('Checkpoint顺序')
        
        # 4. 评估时间分析
        if 'evaluation_time' in df.columns:
            axes[1, 1].bar(range(len(df)), df['evaluation_time'], alpha=0.7, color='lightgreen')
            axes[1, 1].set_xlabel('Checkpoint索引')
            axes[1, 1].set_ylabel('评估时间 (秒)')
            axes[1, 1].set_title('各Checkpoint评估时间')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 添加平均线
            mean_time = df['evaluation_time'].mean()
            axes[1, 1].axhline(mean_time, color='red', linestyle='--', 
                             label=f'平均时间: {mean_time:.2f}s')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("性能分布分析图表已保存")
    
    def correlation_analysis(self, df: pd.DataFrame):
        """相关性分析"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            logger.warning("数值列不足，跳过相关性分析")
            return
        
        # 计算相关性矩阵
        corr_matrix = numeric_df.corr()
        
        # 绘制相关性热图
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('变量相关性热图')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 找出强相关关系
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # 强相关阈值
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if strong_correlations:
            logger.info("发现强相关关系:")
            for corr in strong_correlations:
                logger.info(f"  {corr['var1']} <-> {corr['var2']}: {corr['correlation']:.3f}")
        
        logger.info("相关性分析图表已保存")
        return corr_matrix
    
    def outlier_detection(self, df: pd.DataFrame):
        """异常值检测"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}
        
        for col in numeric_cols:
            if df[col].notna().sum() < 3:  # 需要至少3个有效数据点
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # IQR方法检测异常值
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if not outliers.empty:
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'outlier_values': outliers[col].tolist(),
                    'bounds': (lower_bound, upper_bound)
                }
        
        if outliers_info:
            logger.info("检测到异常值:")
            for col, info in outliers_info.items():
                logger.info(f"  {col}: {info['count']} 个异常值 ({info['percentage']:.1f}%)")
        
        return outliers_info
    
    def model_scaling_analysis(self, df: pd.DataFrame):
        """模型规模分析（如果有多个不同规模的模型）"""
        if 'model_size' not in df.columns or df['model_size'].isna().all():
            logger.warning("缺少模型规模信息，跳过规模分析")
            return
        
        # 按模型规模分组分析
        size_groups = df.groupby('model_size')
        
        if len(size_groups) < 2:
            logger.warning("模型规模种类不足，跳过规模分析")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 模型规模vs性能
        avg_performance = size_groups['val_loss'].mean()
        std_performance = size_groups['val_loss'].std()
        
        x_pos = range(len(avg_performance))
        axes[0].bar(x_pos, avg_performance, yerr=std_performance, capsize=5,
                   alpha=0.7, color='lightblue', edgecolor='black')
        axes[0].set_xlabel('模型隐藏层大小')
        axes[0].set_ylabel('平均验证损失')
        axes[0].set_title('模型规模vs验证性能')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(avg_performance.index)
        axes[0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (mean_val, std_val) in enumerate(zip(avg_performance, std_performance)):
            axes[0].text(i, mean_val + std_val + 0.001, f'{mean_val:.4f}',
                        ha='center', va='bottom')
        
        # 模型规模vs训练效率
        if 'evaluation_time' in df.columns:
            avg_time = size_groups['evaluation_time'].mean()
            axes[1].bar(range(len(avg_time)), avg_time, alpha=0.7, 
                       color='lightcoral', edgecolor='black')
            axes[1].set_xlabel('模型隐藏层大小')
            axes[1].set_ylabel('平均评估时间 (秒)')
            axes[1].set_title('模型规模vs评估效率')
            axes[1].set_xticks(range(len(avg_time)))
            axes[1].set_xticklabels(avg_time.index)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("模型规模分析图表已保存")
    
    def generate_analysis_report(self, df: pd.DataFrame, basic_stats: Dict[str, Any], 
                               outliers_info: Dict[str, Any]) -> str:
        """生成详细分析报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f'analysis_report_{timestamp}.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# RNA-ProGen3 训练结果深度分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**数据样本数**: {len(df)}\n\n")
            
            # 基础统计
            f.write("## 基础统计信息\n\n")
            if 'loss_stats' in basic_stats:
                loss_stats = basic_stats['loss_stats']
                f.write("### 验证损失统计\n")
                f.write(f"- 最小值: {loss_stats['min']:.6f}\n")
                f.write(f"- 最大值: {loss_stats['max']:.6f}\n")
                f.write(f"- 平均值: {loss_stats['mean']:.6f}\n")
                f.write(f"- 标准差: {loss_stats['std']:.6f}\n")
                f.write(f"- 总改善: {loss_stats['improvement']:.6f}\n\n")
            
            if 'perplexity_stats' in basic_stats:
                ppl_stats = basic_stats['perplexity_stats']
                f.write("### 验证困惑度统计\n")
                f.write(f"- 最小值: {ppl_stats['min']:.3f}\n")
                f.write(f"- 最大值: {ppl_stats['max']:.3f}\n")
                f.write(f"- 平均值: {ppl_stats['mean']:.3f}\n")
                f.write(f"- 标准差: {ppl_stats['std']:.3f}\n\n")
            
            # 异常值检测结果
            if outliers_info:
                f.write("## 异常值检测\n\n")
                for col, info in outliers_info.items():
                    f.write(f"### {col}\n")
                    f.write(f"- 异常值数量: {info['count']} ({info['percentage']:.1f}%)\n")
                    f.write(f"- 正常范围: [{info['bounds'][0]:.4f}, {info['bounds'][1]:.4f}]\n")
                    f.write(f"- 异常值: {info['outlier_values']}\n\n")
            else:
                f.write("## 异常值检测\n\n未检测到显著异常值。\n\n")
            
            # 训练趋势分析
            f.write("## 训练趋势分析\n\n")
            if 'val_loss' in df.columns and len(df) > 1:
                initial_loss = df['val_loss'].iloc[0]
                final_loss = df['val_loss'].iloc[-1]
                improvement_pct = (initial_loss - final_loss) / initial_loss * 100
                
                f.write(f"- 初始验证损失: {initial_loss:.6f}\n")
                f.write(f"- 最终验证损失: {final_loss:.6f}\n")
                f.write(f"- 相对改善: {improvement_pct:.2f}%\n")
                
                # 趋势分析
                if len(df) > 2:
                    x = df['step'].values.reshape(-1, 1)
                    y = df['val_loss'].values
                    model = LinearRegression().fit(x, y)
                    slope = model.coef_[0]
                    r_squared = model.score(x, y)
                    
                    f.write(f"- 线性趋势斜率: {slope:.8f}\n")
                    f.write(f"- 拟合度 (R²): {r_squared:.4f}\n")
                    
                    if slope < 0:
                        f.write("- 趋势: 验证损失总体呈下降趋势 ✅\n")
                    else:
                        f.write("- 趋势: 验证损失可能存在上升趋势 ⚠️\n")
            
            f.write("\n")
            
            # 生成的图表
            f.write("## 生成的分析图表\n\n")
            f.write("- `training_dynamics.png`: 训练动态分析（损失曲线、收敛速度等）\n")
            f.write("- `performance_distribution.png`: 性能分布分析\n")
            f.write("- `correlation_heatmap.png`: 变量相关性热图\n")
            f.write("- `model_scaling_analysis.png`: 模型规模分析（如果适用）\n\n")
            
            # 建议
            f.write("## 分析建议\n\n")
            if 'val_loss' in df.columns and len(df) > 1:
                improvement = df['val_loss'].iloc[0] - df['val_loss'].iloc[-1]
                if improvement > 0.01:
                    f.write("✅ 模型显示出良好的学习能力，验证损失显著下降。\n")
                elif improvement > 0:
                    f.write("⚠️ 模型有轻微改善，可能需要更多训练步数或调整超参数。\n")
                else:
                    f.write("❌ 模型未显示明显改善，建议检查学习率、数据质量或模型架构。\n")
            
            if outliers_info:
                f.write("⚠️ 检测到异常值，建议进一步检查对应的checkpoints。\n")
            
            f.write("\n")
        
        logger.info(f"详细分析报告生成: {report_file}")
        return str(report_file)
    
    def run_comprehensive_analysis(self, data_path: str):
        """运行综合分析"""
        logger.info(f"开始综合分析: {data_path}")
        
        # 加载数据
        df = self.load_experiment_data(data_path)
        
        if df.empty:
            logger.error("数据为空，无法进行分析")
            return
        
        # 基础统计
        basic_stats = self.basic_statistics(df)
        
        # 训练动态分析
        self.training_dynamics_analysis(df)
        
        # 性能分布分析
        self.performance_distribution_analysis(df)
        
        # 相关性分析
        self.correlation_analysis(df)
        
        # 异常值检测
        outliers_info = self.outlier_detection(df)
        
        # 模型规模分析
        self.model_scaling_analysis(df)
        
        # 生成综合报告
        report_file = self.generate_analysis_report(df, basic_stats, outliers_info)
        
        logger.info(f"🎉 综合分析完成！")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"分析报告: {report_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='RNA-ProGen3 结果分析')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据文件路径（CSV或JSON格式）')
    parser.add_argument('--output_dir', type=str,
                       help='输出目录')
    parser.add_argument('--config', type=str,
                       help='配置文件路径（可选）')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    config = {
        'log_level': 'INFO'
    }
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    else:
        # 在数据文件同目录下创建analysis目录
        data_dir = Path(args.data_path).parent
        config['output_dir'] = data_dir / 'analysis'
    
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config.update(yaml.safe_load(f))
    
    # 创建分析器并运行分析
    analyzer = ResultAnalyzer(config)
    analyzer.run_comprehensive_analysis(args.data_path)


if __name__ == '__main__':
    main()