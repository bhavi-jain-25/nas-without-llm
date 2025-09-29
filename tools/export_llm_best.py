#!/usr/bin/env python3
import os
import json
import argparse


TEMPLATE_CNN = """
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
"""


TEMPLATE_DEMO = """
import json
import torch

from cnnnet_simple import SimpleCNN


def main():
    with open('best_structure.json', 'r') as f:
        best = json.load(f)

    print('Loaded structure with', len(best.get('structure_info', [])), 'blocks')

    # Build a compact CNN (for CIFAR-10 sized inputs)
    model = SimpleCNN(in_channels=3, num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    print('Output shape:', tuple(y.shape))

if __name__ == '__main__':
    main()
"""


def export(evolution_result: str, output_root: str):
    with open(evolution_result, 'r') as f:
        data = json.load(f)

    best_arch = data['best_architecture']
    tag = 'best_evolved'
    out_dir = os.path.join(output_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Write best structure
    with open(os.path.join(out_dir, 'best_structure.json'), 'w') as f:
        json.dump(best_arch, f, indent=2)

    # Write minimal cnn and demo
    with open(os.path.join(out_dir, 'cnnnet_simple.py'), 'w') as f:
        f.write(TEMPLATE_CNN)
    with open(os.path.join(out_dir, 'demo.py'), 'w') as f:
        f.write(TEMPLATE_DEMO)

    print(f'Exported to {out_dir}')


def parse_args():
    p = argparse.ArgumentParser(description='Export best evolved architecture')
    p.add_argument('--evolution_result', type=str,
                   default='llm_evolution_demo_results/evolution_result.json',
                   help='Path to evolution_result.json')
    p.add_argument('--output_dir', type=str, default='output_llm',
                   help='Root output directory')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export(args.evolution_result, args.output_dir)


