"""Train BiteWise's local EfficientNet-B0 classifier on a class-folder dataset.

Expected layout: data/foodsg233_subset/<food label>/<image files>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).parent


def split_indices(targets: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)
    train_ids, val_ids, test_ids = [], [], []
    for indices in by_class.values():
        shuffled = rng.permutation(indices).tolist()
        n = len(shuffled)
        if n < 3:
            raise ValueError("Every food category needs at least 3 images for train/validation/test splits.")
        n_train = max(1, int(n * .8))
        n_val = max(1, int(n * .1)) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = n - 2
        train_ids.extend(shuffled[:n_train])
        val_ids.extend(shuffled[n_train:n_train + n_val])
        test_ids.extend(shuffled[n_train + n_val:])
    return train_ids, val_ids, test_ids


def make_model(num_classes: int, imagenet_weights: Path | None) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    if imagenet_weights:
        state = torch.load(imagenet_weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        # The classifier has a different size for the local food classes.
        state = {key: value for key, value in state.items() if not key.startswith("classifier.")}
        missing, _ = model.load_state_dict(state, strict=False)
        if not all(name.startswith("classifier.") for name in missing):
            raise ValueError("The supplied ImageNet checkpoint does not match EfficientNet-B0.")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def accuracy_and_macro_f1(predictions: list[int], labels: list[int], class_count: int) -> tuple[float, float]:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for truth, guess in zip(labels, predictions):
        matrix[truth, guess] += 1
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    scores = []
    for i in range(class_count):
        if matrix[i, :].sum() + matrix[:, i].sum() == 0:
            continue
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return accuracy, float(np.mean(scores)) if scores else 0.0


def evaluate(model, loader, device, class_count: int):
    model.eval()
    correct_top3 = total = 0
    predictions: list[int] = []
    labels: list[int] = []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss_total += criterion(output, target).item() * target.size(0)
            top3 = output.topk(min(3, class_count), dim=1).indices
            correct_top3 += (top3 == target[:, None]).any(dim=1).sum().item()
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            labels.extend(target.cpu().tolist())
            total += target.size(0)
    top1, macro_f1 = accuracy_and_macro_f1(predictions, labels, class_count)
    return loss_total / max(total, 1), top1, correct_top3 / max(total, 1), macro_f1


def run_epoch(model, loader, optimizer, device):
    model.train()
    # Frozen blocks must keep their running BatchNorm statistics unchanged.
    for block in model.features:
        if not any(parameter.requires_grad for parameter in block.parameters()):
            block.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "foodsg233_subset")
    parser.add_argument("--imagenet-weights", type=Path, help="Local EfficientNet-B0 ImageNet checkpoint; omit to train from scratch.")
    parser.add_argument("--from-scratch", action="store_true", help="Explicitly train all layers without ImageNet initialization (slower; requires more data).")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--external-test", type=Path, help="Optional independently collected class-folder photos, kept outside the train/validation/test split.")
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.imagenet_weights is None and not args.from_scratch:
        parser.error("Provide --imagenet-weights /path/to/checkpoint.pth for transfer learning, or explicitly choose --from-scratch.")
    if args.imagenet_weights is not None and args.from_scratch:
        parser.error("Choose either --imagenet-weights or --from-scratch.")
    if args.batch_size < 2 or args.head_epochs < 0 or args.finetune_epochs < 1:
        parser.error("Use a batch size of at least 2, nonnegative head epochs, and at least 1 fine-tuning epoch.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not args.data.exists():
        raise SystemExit(f"Dataset folder not found: {args.data}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.72, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.12),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    train_base = datasets.ImageFolder(args.data, transform=train_transform)
    eval_base = datasets.ImageFolder(args.data, transform=eval_transform)
    if len(train_base.classes) < 2:
        raise SystemExit("Add at least two food-category folders with images before training.")
    labels = set(train_base.classes)
    with (ROOT / "data" / "nutrition.csv").open(encoding="utf-8") as handle:
        known = {row["food"] for row in csv.DictReader(handle)}
    unmapped = sorted(labels - known)
    if unmapped:
        raise SystemExit("Add nutrition rows for these classifier labels first: " + ", ".join(unmapped))

    train_ids, val_ids, test_ids = split_indices(train_base.targets, args.seed)
    if not val_ids or not test_ids:
        raise SystemExit("Each dish needs at least three images to create train, validation, and test splits.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} · classes: {len(train_base.classes)} · images: {len(train_base)}")
    print(f"Stratified split: train {len(train_ids)} / validation {len(val_ids)} / test {len(test_ids)}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(train_base, train_ids), batch_size=args.batch_size, shuffle=True, num_workers=2, generator=generator)
    val_loader = DataLoader(Subset(eval_base, val_ids), batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(Subset(eval_base, test_ids), batch_size=args.batch_size, shuffle=False, num_workers=2)
    external_loader = None
    if args.external_test:
        external = datasets.ImageFolder(args.external_test, transform=eval_transform)
        unknown = sorted(set(external.classes) - set(train_base.classes))
        if unknown:
            raise SystemExit("External test includes classes absent from training: " + ", ".join(unknown))
        class_to_idx = train_base.class_to_idx
        external.samples = [(path, class_to_idx[external.classes[old_idx]]) for path, old_idx in external.samples]
        external.targets = [target for _, target in external.samples]
        if not external.samples:
            raise SystemExit("External test folder contains no supported images.")
        external_loader = DataLoader(external, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = make_model(len(train_base.classes), args.imagenet_weights).to(device)

    best_val = -1.0
    best_state = None
    phases = [("head", args.head_epochs), ("fine-tune", args.finetune_epochs)] if not args.from_scratch else [("scratch", args.finetune_epochs)]
    for phase, epochs in phases:
        stale = 0
        if best_state is not None:
            model.load_state_dict(best_state)
        for parameter in model.parameters():
            parameter.requires_grad = phase == "scratch"
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if phase == "fine-tune":
            for parameter in model.features[-2:].parameters():
                parameter.requires_grad = True
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3 if phase == "head" else 1e-4, weight_decay=1e-4)
        for epoch in range(epochs):
            train_loss = run_epoch(model, train_loader, optimizer, device)
            val_loss, val_top1, val_top3, val_f1 = evaluate(model, val_loader, device, len(train_base.classes))
            print(f"{phase} {epoch + 1}/{epochs}: train loss {train_loss:.3f} · val loss {val_loss:.3f} · top-1 {val_top1:.3f} · top-3 {val_top3:.3f} · macro F1 {val_f1:.3f}")
            if val_top1 > best_val:
                best_val = val_top1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 3:
                print("Early stopping: validation accuracy has not improved for 3 epochs.")
                break

    if best_state is None:
        raise SystemExit("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    test_loss, test_top1, test_top3, test_f1 = evaluate(model, test_loader, device, len(train_base.classes))
    args.output.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "seed": args.seed,
        "classes": train_base.classes,
        "splits": {name: [str(Path(train_base.samples[i][0]).relative_to(args.data)) for i in ids]
                   for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids))},
    }
    (args.output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state_dict": best_state}, args.output / "food_classifier.pt")
    (args.output / "class_names.json").write_text(json.dumps(train_base.classes, indent=2) + "\n", encoding="utf-8")
    metrics = {"test_loss": test_loss, "top1_accuracy": test_top1, "top3_accuracy": test_top3, "macro_f1": test_f1, "classes": train_base.classes}
    if external_loader is not None:
        ext_loss, ext_top1, ext_top3, ext_f1 = evaluate(model, external_loader, device, len(train_base.classes))
        metrics["external_test"] = {"loss": ext_loss, "top1_accuracy": ext_top1, "top3_accuracy": ext_top3, "macro_f1": ext_f1}
        print(f"Independent photo set: top-1 {ext_top1:.3f} · top-3 {ext_top3:.3f} · macro F1 {ext_f1:.3f}")
    (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Held-out test: top-1 {test_top1:.3f} · top-3 {test_top3:.3f} · macro F1 {test_f1:.3f}")
    print(f"Saved local model and class names to {args.output}")


if __name__ == "__main__":
    main()
