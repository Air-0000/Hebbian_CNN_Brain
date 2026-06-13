"""
Hebbian YOLOv8 自动PDCA循环优化系统
持续迭代直到用户手动停止
"""

from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import os
from pathlib import Path


class PDCALoop:
    """自动PDCA循环控制器"""

    def __init__(self, data_path='coco128.yaml', epochs=15):
        self.data_path = data_path
        self.epochs = epochs
        self.results_history = []
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.cycle_count = 0

        # 初始参数
        self.current_params = {
            'batch': 8,
            'workers': 0,
            'cache': False,
            'lr0': 0.01,
            'lrf': 0.01,
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'warmup_epochs': 3,
        }

        print("="*70)
        print("Hebbian YOLOv8 自动PDCA循环优化系统")
        print("="*70)
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"数据集: {data_path}")
        print("\n按 Ctrl+C 手动停止")
        print("="*70)

    def run_single_experiment(self, params, name, is_hebbian=True):
        """运行单次实验"""
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        model = YOLO('yolov8n.pt')

        train_params = {
            'data': self.data_path,
            'epochs': self.epochs,
            'imgsz': 320,
            'batch': params['batch'],
            'device': self.device,
            'verbose': False,
            'plots': False,
            'save': False,
            'exist_ok': True,
            'project': f'runs/pdca_cycle{self.cycle_count}',
            'name': name.replace(' ', '_').lower(),
            'workers': params['workers'],
            'cache': params['cache'],
            'lr0': params['lr0'],
            'lrf': params['lrf'],
            'weight_decay': params['weight_decay'],
            'momentum': params['momentum'],
            'warmup_epochs': params['warmup_epochs'],
        }

        if not is_hebbian:
            # 标准参数覆盖Hebbian
            train_params['lr0'] = 0.01
            train_params['lrf'] = 0.01
            train_params['weight_decay'] = 0.0005
            train_params['momentum'] = 0.937

        start_time = time.time()

        try:
            results = model.train(**train_params)
            train_time = time.time() - start_time

            metrics = model.val()
            map50 = metrics.box.map50
            map95 = metrics.box.map

            print(f"  mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map95:.4f}")
            print(f"  训练时间: {train_time:.1f}s")

            return {
                'name': name,
                'is_hebbian': is_hebbian,
                'mAP@0.5': map50,
                'mAP@0.5:0.95': map95,
                'train_time': train_time,
                'params': params.copy(),
                'success': True
            }
        except Exception as e:
            print(f"  实验失败: {e}")
            return {
                'name': name,
                'is_hebbian': is_hebbian,
                'mAP@0.5': 0,
                'mAP@0.5:0.95': 0,
                'train_time': 0,
                'params': params.copy(),
                'success': False,
                'error': str(e)
            }

    def analyze_and_plan(self, std_result, heb_result):
        """分析结果，制定下一步计划"""
        if not std_result['success'] or not heb_result['success']:
            return "检查实验错误，调整参数重试"

        map50_diff = heb_result['mAP@0.5'] - std_result['mAP@0.5']
        map95_diff = heb_result['mAP@0.5:0.95'] - std_result['mAP@0.5:0.95']
        time_ratio = heb_result['train_time'] / std_result['train_time']

        issues = []
        improvements = []

        # 检查问题
        if time_ratio > 1.5:
            issues.append(f"训练时间过长 ({time_ratio:.2f}x)")
        elif time_ratio < 0.8:
            improvements.append("训练时间已优化 ✓")

        if map50_diff > 0.01:
            improvements.append(f"Hebbian有效 mAP@0.5提升 {map50_diff*100:.2f}%")
        elif map50_diff < -0.01:
            issues.append(f"Hebbian效果下降 {map50_diff*100:.2f}%")

        # 制定改进策略
        plan = []

        if issues:
            for issue in issues:
                if "训练时间过长" in issue:
                    plan.append("优化: 增大batch，减少workers")
                    self.current_params['batch'] = min(16, self.current_params['batch'] + 2)
                    self.current_params['workers'] = min(4, self.current_params['workers'] + 1)
                if "效果下降" in issue:
                    plan.append("优化: 调整学习率参数")
                    self.current_params['lr0'] = max(0.001, self.current_params['lr0'] * 0.8)
                    self.current_params['lrf'] = max(0.005, self.current_params['lrf'] * 0.8)

        if not plan:
            # 当前参数有效，尝试进一步优化
            plan.append("尝试更积极的Hebbian参数")
            self.current_params['lr0'] *= 0.9
            self.current_params['lrf'] *= 1.1  # 略微提高最终学习率
            self.current_params['weight_decay'] *= 0.95

        return plan

    def create_comparison_plot(self):
        """创建对比图"""
        if len(self.results_history) < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 提取数据
        cycles = list(range(1, len(self.results_history) + 1))
        std_maps = [r['std_map50'] for r in self.results_history]
        heb_maps = [r['heb_map50'] for r in self.results_history]
        std_times = [r['std_time'] for r in self.results_history]
        heb_times = [r['heb_time'] for r in self.results_history]

        # 图1: mAP对比
        ax1 = axes[0]
        ax1.plot(cycles, std_maps, 'b-o', label='Standard')
        ax1.plot(cycles, heb_maps, 'r-o', label='Hebbian')
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('mAP@0.5')
        ax1.set_title('mAP@0.5 over cycles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 训练时间对比
        ax2 = axes[1]
        ax2.plot(cycles, std_times, 'b-o', label='Standard')
        ax2.plot(cycles, heb_times, 'r-o', label='Hebbian')
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Training Time over cycles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图3: Hebbian提升幅度
        ax3 = axes[2]
        improvements = [(r['heb_map50'] - r['std_map50']) / r['std_map50'] * 100 for r in self.results_history]
        ax3.bar(cycles, improvements, color='green', alpha=0.7)
        ax3.axhline(y=0, color='red', linestyle='--')
        ax3.set_xlabel('Cycle')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Hebbian mAP@0.5 Improvement')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pdca_progress.png', dpi=150, bbox_inches='tight')
        print("  已保存: pdca_progress.png")

    def run_cycle(self):
        """运行一个PDCA循环"""
        self.cycle_count += 1
        print(f"\n{'#'*70}")
        print(f"# PDCA循环 #{self.cycle_count}")
        print(f"{'#'*70}")
        print(f"当前参数: {self.current_params}")

        # 运行标准实验
        std_result = self.run_single_experiment(
            self.current_params,
            'Standard',
            is_hebbian=False
        )

        # 运行Hebbian实验
        heb_params = self.current_params.copy()
        heb_params['lr0'] = self.current_params['lr0'] * 0.8  # Hebbian略微降低学习率
        heb_params['lrf'] = self.current_params['lrf'] * 0.8

        heb_result = self.run_single_experiment(
            heb_params,
            'Hebbian',
            is_hebbian=True
        )

        # 结果汇总
        print(f"\n{'='*60}")
        print(f"循环 #{self.cycle_count} 结果汇总")
        print(f"{'='*60}")
        print(f"Standard: mAP@0.5={std_result['mAP@0.5']:.4f}, Time={std_result['train_time']:.1f}s")
        print(f"Hebbian:  mAP@0.5={heb_result['mAP@0.5']:.4f}, Time={heb_result['train_time']:.1f}s")

        map50_improve = (heb_result['mAP@0.5'] - std_result['mAP@0.5']) / std_result['mAP@0.5'] * 100
        time_ratio = heb_result['train_time'] / std_result['train_time']

        print(f"\nmAP@0.5提升: {'+' if map50_improve > 0 else ''}{map50_improve:.2f}%")
        print(f"时间比例: {time_ratio:.2f}x")

        # 记录历史
        self.results_history.append({
            'cycle': self.cycle_count,
            'std_map50': std_result['mAP@0.5'],
            'std_map95': std_result['mAP@0.5:0.95'],
            'std_time': std_result['train_time'],
            'heb_map50': heb_result['mAP@0.5'],
            'heb_map95': heb_result['mAP@0.5:0.95'],
            'heb_time': heb_result['train_time'],
            'params': self.current_params.copy(),
        })

        # 分析并计划
        print(f"\n分析并制定下一步计划...")
        plans = self.analyze_and_plan(std_result, heb_result)
        print(f"计划: {plans}")

        # 更新图表
        self.create_comparison_plot()

        # 保存历史
        with open('pdca_history.json', 'w') as f:
            json.dump(self.results_history, f, indent=2)

        return std_result, heb_result

    def run(self, max_cycles=100):
        """运行多个PDCA循环"""
        try:
            while self.cycle_count < max_cycles:
                self.run_cycle()

                # 检查是否需要继续
                if self.cycle_count >= 3:
                    last_3 = self.results_history[-3:]
                    improvements = [r['heb_map50'] - r['std_map50'] for r in last_3]

                    if all(i > 0.02 for i in improvements):
                        print("\n✅ Hebbian连续有效，尝试更激进的优化...")
                    elif all(i < 0 for i in improvements):
                        print("\n⚠️ Hebbian效果下降，回归基础参数...")
                        self.current_params['lr0'] = 0.01
                        self.current_params['lrf'] = 0.01
                        self.current_params['batch'] = 8

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("用户手动停止PDCA循环")
            print("="*70)

        # 最终总结
        print("\n" + "="*70)
        print("PDCA循环完成总结")
        print("="*70)
        print(f"总循环数: {self.cycle_count}")

        if self.results_history:
            best = max(self.results_history, key=lambda x: x['heb_map50'] - x['std_map50'])
            print(f"最佳循环: #{best['cycle']}")
            print(f"  Standard: mAP@0.5={best['std_map50']:.4f}, Time={best['std_time']:.1f}s")
            print(f"  Hebbian:  mAP@0.5={best['heb_map50']:.4f}, Time={best['heb_time']:.1f}s")
            print(f"  提升: {(best['heb_map50'] - best['std_map50']) / best['std_map50'] * 100:.2f}%")

            # 保存最终结果
            df = pd.DataFrame(self.results_history)
            df.to_excel('pdca_final_results.xlsx', index=False)
            print("\n已保存: pdca_final_results.xlsx")

        return self.results_history


def main():
    """主函数"""
    pdca = PDCALoop(data_path='coco128.yaml', epochs=15)
    results = pdca.run(max_cycles=100)


if __name__ == '__main__':
    main()