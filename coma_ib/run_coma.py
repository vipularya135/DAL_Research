import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from models import VGG16
from coma_ib.coma import coma_select_samples
from coma_ib.ib_score import compute_ib_scores

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return train

def run_coma_experiment(k=500):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_cifar10()

    # Two models for co-training
    model_a = VGG16(num_classes=10).to(device)
    model_b = VGG16(num_classes=10).to(device)

    print("\nRunning coMA Sample Selection...")
    coma_indices = coma_select_samples(model_a, model_b, dataset, k=k, device=device)

    print(f"coMA selected {len(coma_indices)} samples.")

    print("\nRunning IB Scoring...")
    ib_indices = compute_ib_scores(model_a, dataset, k=k, device=device)

    print(f"IB selected {len(ib_indices)} samples.")

    # Merge both lists (unique)
    merged = list(set(coma_indices + ib_indices))

    print(f"\nTotal selected (merged coMA + IB): {len(merged)} samples")

    # Return dataset subset
    selected_dataset = Subset(dataset, merged)
    return selected_dataset, merged


if __name__ == "__main__":
    selected, idx = run_coma_experiment()
    print("\nSample Indices:", idx[:20])
