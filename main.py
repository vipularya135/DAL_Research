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

    print("\nRunning LCD method on VGG16 with CIFAR-10...\n")

  
    exp_acc = train_until_empty(
        model,
        initial_train_set,
        remainder,
        test_set,
        epochs=50,
        max_iterations=20,
        batch_size=32,
        learning_rate=0.01,
        method=1,
        run_tag="lcd_vgg16_cifar10"
    )

    print("Final Accuracies:", exp_acc)

main()
