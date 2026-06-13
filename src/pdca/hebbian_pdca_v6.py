"""
Hebbian YOLOv8 PDCA优化版 v6
解决：稳定性问题、更短训练周期
"""

from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import os


class PDCALoopV6:
    def __init__(self, data_path='coco128.yaml', epochs=8):
        self.data_path = data_path
        self.epochs = epochs
        self.device = 'mps'
        self.cycle_count = 0
        self.best_improve = -1
        self.best_params = None
        self.history = []

        # 简化参数提高稳定性
        self.params = {
            'batch': 16,
            'workers': 2,
            'cache': False,
            'lr0': 0.008,
            'lrf': 0.015,
            'weight_decay': 0.0004,
            'warmup_epochs': 2,
        }

        print("="*70)
        print("Hebbian YOLOv8 PDCA v6 - 稳定版")
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
            'project': f'runs/v6_cycle{self.cycle_count}',
            'name': name.replace(' ', '_'),
            'workers': params['workers'],
            'cache': params['cache'],
            'warmup_epochs': params['warmup_epochs'],
            'cos_lr': False,  # 禁用cos_lr提高稳定性
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
        try:
            results = model.train(**train_params)
            train_time = time.time() - start
            metrics = model.val()
            map50 = metrics.box.map50
            map95 = metrics.box.map
            print(f"  mAP@0.5={map50:.4f}, time={train_time:.1f}s")
            return {'mAP@0.5': map50, 'mAP@0.5:0.95': map95, 'time': train_time}
        except Exception as e:
            print(f"  训练失败: {e}")
            return {'mAP@0.5': 0, 'mAP@0.5:0.95': 0, 'time': 0}

    def analyze(self, std, heb):
        """分析结果"""
        if std['mAP@0.5'] == 0:
            return {'improve': 0, 'time_ratio': 0}
        improve = (heb['mAP@0.5'] - std['mAP@0.5']) / std['mAP@0.5'] * 100
        time_ratio = heb['time'] / std['time'] if std['time'] > 0 else 0

        print(f"\n📊 分析:")
        print(f"  mAP提升: {'+' if improve > 0 else ''}{improve:.3f}%")
        print(f"  时间比例: {time_ratio:.2f}x")

        if improve > self.best_improve:
            self.best_improve = improve
            self.best_params = self.params.copy()

        if improve > 0.15 and time_ratio < 1.3:
            print("  ✅ 效果优秀")
        elif time_ratio > 1.5:
            print("  ⚠️ 训练时间过长")
            self.params['batch'] = min(20, self.params['batch'] + 2)
        elif improve < -0.2:
            print("  ⚠️ Hebbian效果下降")
            self.params['lr0'] = 0.01
        else:
            print("  🔄 继续微调")
            self.params['lr0'] *= 0.95

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

        with open('pdca_v6_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return std, heb

    def run(self, max_cycles=15):
        try:
            while self.cycle_count < max_cycles:
                self.run_cycle()

                if len(self.history) >= 3:
                    recent = self.history[-3:]
                    if all(r['improve'] < -0.5 for r in recent):
                        print("\n⚠️ 连续3次效果下降，终止PDCA")
                        break

        except KeyboardInterrupt:
            print("\n\n用户停止")

        self.summary()

    def summary(self):
        print("\n" + "="*70)
        print("PDCA v6 完成总结")
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

            df = pd.DataFrame([{
                'cycle': h['cycle'],
                'std_map50': h['std']['mAP@0.5'],
                'std_time': h['std']['time'],
                'heb_map50': h['heb']['mAP@0.5'],
                'heb_time': h['heb']['time'],
                'improve': h['improve'],
                'time_ratio': h['time_ratio']
            } for h in self.history])
            df.to_excel('pdca_v6_results.xlsx', index=False)
            print("\n已保存: pdca_v6_results.xlsx")


if __name__ == '__main__':
    pdca = PDCALoopV6(data_path='coco128.yaml', epochs=8)
    pdca.run(max_cycles=15)