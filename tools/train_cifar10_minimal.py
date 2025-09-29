#!/usr/bin/env python3
"""
Minimal CIFAR-10 training harness for the exported SimpleCNN model.
Intended for a quick sanity-check training loop (not tuned for accuracy).
"""

import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import importlib.util


def load_model(cnn_file: str):
    spec = importlib.util.spec_from_file_location('cnnnet_simple', cnn_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SimpleCNN(in_channels=3, num_classes=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export_dir', type=str, default='output_llm/best_evolved',
                   help='Directory containing cnnnet_simple.py and best_structure.json')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--save_dir', type=str, default='runs/cifar10_minimal',
                   help='Directory to save checkpoints and metrics')
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--mixup', type=float, default=0.0, help='MixUp alpha; 0 disables')
    p.add_argument('--cutmix', type=float, default=0.0, help='CutMix alpha; 0 disables')
    p.add_argument('--cosine', action='store_true', help='Use cosine LR schedule')
    p.add_argument('--amp', action='store_true', help='Enable CUDA AMP mixed precision')
    p.add_argument('--compile', action='store_true', help='Use torch.compile for the model')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.workers > 0,
    )

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=args.pin_memory,
        persistent_workers=args.workers > 0,
    )

    # Model
    model_path = os.path.join(args.export_dir, 'cnnnet_simple.py')
    model = load_model(model_path).to(device)

    if args.compile and device.type == 'cuda':
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.cosine:
        lr_min = args.lr * 1e-2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr_min)
    else:
        scheduler = None

    use_cuda_amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = math.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        bbx1 = max(cx - cut_w // 2, 0)
        bby1 = max(cy - cut_h // 2, 0)
        bbx2 = min(cx + cut_w // 2, W)
        bby2 = min(cy + cut_h // 2, H)
        return bbx1, bby1, bbx2, bby2

    def one_hot(labels, num_classes):
        return torch.zeros((labels.size(0), num_classes), device=labels.device).scatter_(1, labels.unsqueeze(1), 1.0)

    def soft_cross_entropy(logits, target_probs):
        log_probs = torch.log_softmax(logits, dim=1)
        return -(target_probs * log_probs).sum(dim=1).mean()

    os.makedirs(args.save_dir, exist_ok=True)

    def evaluate():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100.0 * correct / total

    # Train
    best_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start = time.time()
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            do_mix = (args.mixup > 0.0 or args.cutmix > 0.0)
            mixed_target = None

            if do_mix:
                if args.cutmix > 0.0 and (args.mixup == 0.0 or torch.rand(1).item() < 0.5):
                    lam = torch.distributions.Beta(args.cutmix, args.cutmix).sample().item()
                    rand_index = torch.randperm(images.size(0), device=images.device)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
                    y1 = one_hot(labels, 10)
                    y2 = one_hot(labels[rand_index], 10)
                    mixed_target = lam * y1 + (1 - lam) * y2
                else:
                    lam = torch.distributions.Beta(args.mixup, args.mixup).sample().item()
                    rand_index = torch.randperm(images.size(0), device=images.device)
                    mixed_images = lam * images + (1 - lam) * images[rand_index, :]
                    y1 = one_hot(labels, 10)
                    y2 = one_hot(labels[rand_index], 10)
                    mixed_target = lam * y1 + (1 - lam) * y2
                    images = mixed_images

            with torch.cuda.amp.autocast(enabled=use_cuda_amp):
                outputs = model(images)
                if mixed_target is not None:
                    loss = soft_cross_entropy(outputs, mixed_target)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch+1} Iter {i+1}: loss={running_loss / (i+1):.4f}')
        dt = time.time() - start
        acc = evaluate()
        print(f'Epoch {epoch+1} done in {dt:.1f}s, test acc {acc:.2f}%')
        history.append({
            'epoch': int(epoch + 1),
            'train_time_s': float(dt),
            'test_acc': float(acc),
            'lr': float(optimizer.param_groups[0]['lr'])
        })
        # Save best checkpoint
        if acc > best_acc:
            best_acc = acc
            ckpt_path = os.path.join(args.save_dir, 'best.pth')
            torch.save({'model_state': model.state_dict(),
                        'best_acc': best_acc,
                        'epoch': epoch + 1,
                        'args': vars(args)}, ckpt_path)
        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

    # Save metrics JSON
    try:
        import json
        metrics = {
            'best_acc': float(best_acc),
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'label_smoothing': float(args.label_smoothing),
            'mixup': float(args.mixup),
            'cutmix': float(args.cutmix),
            'cosine': bool(args.cosine),
            'amp': bool(args.amp),
            'device': device.type,
            'history': history,
        }
        with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass


if __name__ == '__main__':
    main()


