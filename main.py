# main.py
from models import *
from swin import *
from vit_tiny import *
from func_def import prepare_data, train_until_empty

import torch

def main():
    DATASET = "cifar10"
    SEED = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(SEED)

    # Load dataset
    initial_train_set, remainder, test_set = prepare_data(DATASET)

    # Choose model
    model = VGG16(num_classes=10).to(device)

    print("\nRunning EWC experiment (method=5) on VGG16...\n")

    # Run EWC-enabled training
    exp_acc = train_until_empty(
        model,
        initial_train_set,
        remainder,
        test_set,
        epochs=50,
        max_iterations=20,
        batch_size=32,
        learning_rate=0.01,
        method=5,                          # Method 5 = EWC
        run_tag="ewc_vgg16_cifar10",
        ewc_lambda=1.0,                    # Recommended EWC strength
        ewc_samples_for_fisher=20          # Reasonable Fisher batches
    )

    print("Final Accuracies:", exp_acc)

if __name__ == "__main__":
    main()# main.py
from models import *
from swin import *
from vit_tiny import *
from func_def import prepare_data, train_until_empty

import torch

def main():
    DATASET = "cifar10"
    SEED = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(SEED)

    initial_train_set, remainder, test_set = prepare_data(DATASET)

    model = VGG16(num_classes=10).to(device)

    print("\nRunning EWC experiment (method=5) on VGG16...\n")

    exp_acc = train_until_empty(
        model,
        initial_train_set,
        remainder,
        test_set,
        epochs=50,
        max_iterations=20,
        batch_size=32,
        learning_rate=0.01,
        method=5,                                   # 5 -> EWC
        run_tag="ewc_vgg16_cifar10_" + DATASET,
        ewc_lambda=1000.0,                          # main param name
        lambda_ewc=1000.0                           # alias (keeps compatibility)
    )

    print("Final Accuracies:", exp_acc)

if __name__ == "__main__":
    main()
