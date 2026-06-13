"""
Hebbian YOLOv8 PDCA优化版 v4
问题：训练时间过长（Hebbian 3.6x Standard）
解决：更稳定的学习率策略
"""

from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import os


class PDCALoopV4:
    def __init__(self, data_path='coco128.yaml', epochs=10):
        self.data_path = data_path
        self.epochs = epochs
        self.device = 'mps'
        self.cycle_count = 0
        self.best_improve = -1
        self.best_params = None
        self.history = []

        # 优化后的参数
        self.params = {
            'batch': 12,
            'workers': 2,
            'cache': True,
            'lr0': 0.008,
            'lrf': 0.015,
            'weight_decay': 0.0004,
            'warmup_epochs': 2,
        }

        print("="*70)
        print("Hebbian YOLOv8 PDCA v4 - 优化训练时间")
        print("="*70)
        print(f"设备: {self.device}, epochs: {epochs}")
        print("初始参数:", self.params)
        print("\n按 Ctrl+C 停止")
        print("="*70)

    def train_model(self, params, name, is_hebbian=True):
        print(f"\n--- {name} ---")
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
            'project': f'runs/v4_cycle{self.cycle_count}',
            'name': name.replace(' ', '_'),
            'workers': params['workers'],
            'cache': params['cache'],
            'warmup_epochs': params['warmup_epochs'],
        }

        if is_hebbian:
            train_params.update({
                'lr0': params['lr0'] * 0.85,
                'lrf': params['lrf'] * 0.85,
                'weight_decay': params['weight_decay'] * 0.9,
            })
        else:
            train_params.update({
                'lr0': 0.01,
                'lrf': 0.01,
                'weight_decay': 0.0005,
            })

        start = time.time()
        results = model.train(**train_params)
        train_time = time.time() - start

        metrics = model.val()
        map50 = metrics.box.map50
        map95 = metrics.box.map

        print(f"  mAP@0.5={map50:.4f}, time={train_time:.1f}s")
        return {'mAP@0.5': map50, 'mAP@0.5:0.95': map95, 'time': train_time}

    def analyze(self, std, heb):
        """分析结果，决定下一步"""
        improve = (heb['mAP@0.5'] - std['mAP@0.5']) / std['mAP@0.5'] * 100
        time_ratio = heb['time'] / std['time']

        print(f"\n📊 分析:")
        print(f"  mAP提升: {'+' if improve > 0 else ''}{improve:.3f}%")
        print(f"  时间比例: {time_ratio:.2f}x")

        # 更新最佳
        if improve > self.best_improve:
            self.best_improve = improve
            self.best_params = self.params.copy()

        # 决策
        if improve > 0.15 and time_ratio < 1.3:
            print("  ✅ 效果优秀，保持参数继续优化")
        elif time_ratio > 1.5:
            print("  ⚠️ 训练时间过长，调整参数...")
            self.params['batch'] = min(16, self.params['batch'] + 2)
            self.params['warmup_epochs'] = max(1, self.params['warmup_epochs'] - 1)
            self.params['lr0'] *= 0.9
        elif improve < -0.2:
            print("  ⚠️ Hebbian效果下降，回归基础")
            self.params['lr0'] = 0.01
            self.params['lrf'] = 0.01
        else:
            print("  🔄 继续微调参数")
            self.params['lr0'] *= 0.95
            self.params['lrf'] *= 0.95

        return {'improve': improve, 'time_ratio': time_ratio}

    def run_cycle(self):
        self.cycle_count += 1
        print(f"\n{'#'*60}")
        print(f"# PDCA循环 #{self.cycle_count}")
        print(f"# 参数: {self.params}")
        print(f"{'#'*60}")

        std = self.train_model(self.params, 'Standard', is_hebbian=False)
        heb = self.train_model(self.params, 'Hebbian', is_hebbian=True)

        result = self.analyze(std, heb)

        self.history.append({
            'cycle': self.cycle_count,
            'std': std,
            'heb': heb,
            'params': self.params.copy(),
            'improve': result['improve'],
            'time_ratio': result['time_ratio']
        })

        # 保存
        with open('pdca_v4_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return std, heb

    def run(self, max_cycles=20):
        try:
            while self.cycle_count < max_cycles:
                self.run_cycle()

                # 连续3次效果下降则终止
                if len(self.history) >= 3:
                    recent = self.history[-3:]
                    if all(r['improve'] < 0 for r in recent):
                        print("\n⚠️ 连续3次效果下降，终止PDCA")
                        break

        except KeyboardInterrupt:
            print("\n\n用户停止")

        # 总结
        self.summary()

    def summary(self):
        print("\n" + "="*70)
        print("PDCA v4 完成总结")
        print("="*70)

        if self.history:
            best = max(self.history, key=lambda x: x['improve'])
            print(f"最佳效果: 循环#{best['cycle']}")
            print(f"  mAP提升: {'+' if best['improve'] > 0 else ''}{best['improve']:.3f}%")
            print(f"  时间比例: {best['time_ratio']:.2f}x")

            avg_improve = sum(h['improve'] for h in self.history) / len(self.history)
            avg_time = sum(h['time_ratio'] for h in self.history) / len(self.history)
            print(f"\n平均提升: {'+' if avg_improve > 0 else ''}{avg_improve:.3f}%")
            print(f"平均时间: {avg_time:.2f}x")

            # 保存
            df = pd.DataFrame([{
                'cycle': h['cycle'],
                'std_map50': h['std']['mAP@0.5'],
                'std_time': h['std']['time'],
                'heb_map50': h['heb']['mAP@0.5'],
                'heb_time': h['heb']['time'],
                'improve': h['improve'],
                'time_ratio': h['time_ratio']
            } for h in self.history])
            df.to_excel('pdca_v4_results.xlsx', index=False)
            print("\n已保存: pdca_v4_results.xlsx")


if __name__ == '__main__':
    pdca = PDCALoopV4(data_path='coco128.yaml', epochs=10)
    pdca.run(max_cycles=20)