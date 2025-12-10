# main.py
from models import *
from func_def import prepare_data, train_until_empty

import torch

def main():
    DATASET = "cifar10"
    SEED = 42

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(SEED)

    # Load dataset
    initial_train_set, remainder, test_set = prepare_data(DATASET)

    # Choose model (VGG16 for CIFAR10)
    model = VGG16(num_classes=10).to(device)

    print("\nRunning coMA + IB experiment (method = 6) on VGG16...\n")

    # Run method=6 (coMA + IB)
    exp_acc = train_until_empty(
        model,
        initial_train_set,
        remainder,
        test_set,
        epochs=50,
        max_iterations=20,
        batch_size=32,
        learning_rate=0.01,
        method=6,                        # << coMA + IB
        run_tag="coma_ib_vgg16_cifar10"
    )

    print("Final Accuracies:", exp_acc)


if __name__ == "__main__":
    main()