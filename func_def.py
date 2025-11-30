import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import math
from torchvision.datasets import VOCDetection
import copy

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(dataset_name):
    print('==> Preparing data..')
    
    if dataset_name == 'cifar10':
        # CIFAR-10 Normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'cifar100':
        # CIFAR-100 Normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    elif dataset_name == 'svhn':
        # SVHN Normalization
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    elif dataset_name == 'voc2012':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        train_set = VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)
        test_set = VOCDetection(root='./data', year='2012', image_set='val', download=True, transform=transform)

        
    else:
        raise ValueError("Invalid dataset name. Choose from 'cifar10', 'cifar100', 'svhn', or 'voc2012'.")

    torch.manual_seed(42)
    initial_size = int(0.04 * len(train_set))
    remainder_size = len(train_set) - initial_size
    initial_train_set, remainder = torch.utils.data.random_split(train_set, [initial_size, remainder_size])

    print(f"Size of initial_train_set: {len(initial_train_set)}")
    print(f"Size of remainder: {len(remainder)}")

    return initial_train_set, remainder, test_set


        
def train_model(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
     
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        
        # Step the scheduler
        scheduler.step()
    


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def calculate_cluster_centers(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers

def get_most_diverse_samples(tsne_results, cluster_centers, num_diverse_samples):
    distances = euclidean_distances(tsne_results, cluster_centers)
    min_distances = np.max(distances, axis=1)
    sorted_indices = np.argsort(min_distances)
    diverse_indices = sorted_indices[:num_diverse_samples]
    return diverse_indices

def extract_embeddings(model, test):
    test_loader = data.DataLoader(test, batch_size=64, shuffle=False)
    model.eval()
    embeddings = []
    targets_list = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            intermediate_features = model(images)
            embeddings.extend(intermediate_features.view(intermediate_features.size(0), -1).tolist())
            targets_list.append(targets)
    return embeddings

def least_confidence_images(model, test_dataset, k=None):
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    confidences = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
            labels.extend(targets.cpu().tolist())
    confidences = torch.tensor(confidences)
    k = min(k, len(confidences)) if k is not None else len(confidences)
    _, indices = torch.topk(confidences, k, largest=False)
    return data.Subset(test_dataset, indices), indices.tolist()  # Convert to list

def high_confidence_images(model, test_dataset, k=None):
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    confidences = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
            labels.extend(targets.cpu().tolist())
    confidences = torch.tensor(confidences)
    k = min(k, len(confidences)) if k is not None else len(confidences)
    _, indices = torch.topk(confidences, k, largest=True)
    return data.Subset(test_dataset, indices), indices.tolist()  # Convert to list
    
def HC_diverse(embed_model, remainder, n=None):
    high_conf_images, high_conf_indices = high_confidence_images(embed_model, remainder, k=min(2*n, len(remainder)) if n else len(remainder))
    high_conf_embeddings = extract_embeddings(embed_model, high_conf_images)
    high_conf_embeddings = np.array([np.array(e) for e in high_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(high_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    diverse_images = data.Subset(high_conf_images, diverse_indices)
    return diverse_images, [high_conf_indices[i] for i in diverse_indices]

def LC_diverse(embed_model, remainder, n=None):
    least_conf_images, least_conf_indices = least_confidence_images(embed_model, remainder, k=min(2*n, len(remainder)) if n else len(remainder))
    least_conf_embeddings = extract_embeddings(embed_model, least_conf_images)
    least_conf_embeddings = np.array([np.array(e) for e in least_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(least_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    diverse_images = data.Subset(least_conf_images, diverse_indices)
    return diverse_images, [least_conf_indices[i] for i in diverse_indices]

def LC_HC(model, remainder, n=None):
    least_confident, least_confident_indices = least_confidence_images(model, remainder, k=n//2)
    most_confident, most_confident_indices = high_confidence_images(model, remainder, k=n//2)
    combined_dataset = data.ConcatDataset([least_confident, most_confident])
    combined_indices = least_confident_indices + most_confident_indices  # Now lists are concatenated
    return combined_dataset, combined_indices

def LC_HC_diverse(embed_model, remainder, n, low_conf_ratio=0.5, high_conf_ratio=0.5):
    assert low_conf_ratio + high_conf_ratio == 1.0, "Ratios must sum to 1.0"

    n_low = int(n * low_conf_ratio)
    n_high = int(n * high_conf_ratio)

    # Process low confidence
    least_conf_images, least_conf_indices = least_confidence_images(embed_model, remainder, k=min(2*n_low, len(remainder)))
    least_conf_embeddings = extract_embeddings(embed_model, least_conf_images)
    least_conf_embeddings = np.array([np.array(e) for e in least_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(least_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_low_indices = get_most_diverse_samples(tsne_results, cluster_centers, n_low)
    diverse_least_conf_images = data.Subset(least_conf_images, diverse_low_indices)

    # Process high confidence
    high_conf_images, high_conf_indices = high_confidence_images(embed_model, remainder, k=min(2*n_high, len(remainder)))
    high_conf_embeddings = extract_embeddings(embed_model, high_conf_images)
    high_conf_embeddings = np.array([np.array(e) for e in high_conf_embeddings])
    tsne_results = tsne.fit_transform(high_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_high_indices = get_most_diverse_samples(tsne_results, cluster_centers, n_high)
    diverse_high_conf_images = data.Subset(high_conf_images, diverse_high_indices)

    combined_dataset = data.ConcatDataset([diverse_least_conf_images, diverse_high_conf_images])
    combined_indices = [least_conf_indices[i] for i in diverse_low_indices] + [high_conf_indices[i] for i in diverse_high_indices]

    return combined_dataset, combined_indices

# 5% of new remaining data added to each iteration
# def train_until_empty(model, initial_train_set, remainder_set, test_set, epochs=1, max_iterations=15, batch_size=64, learning_rate=0.01, method=1):
#     exp_acc = []
    
#     for iteration in range(max_iterations):
#         print(f"Starting Iteration {iteration+1}")
#         print(f"Remainder Size: {len(remainder_set)}")
#         if len(remainder_set) == 0:
#             print("Dataset empty. Stopping.")
#             break
                    
#         n_samples = int(0.05 * len(remainder_set))
#         n_samples = min(n_samples, len(remainder_set)) 

#         if method == 1:
#             train_data, used_indices = LC_HC(model, remainder_set, n=n_samples)
#         elif method == 2:
#             train_data, used_indices = LC_HC_diverse(model, remainder_set, n=n_samples)
#         elif method == 3:
#             train_data, used_indices = HC_diverse(model, remainder_set, n=n_samples)
#         elif method == 4:
#             train_data, used_indices = LC_diverse(model, remainder_set, n=n_samples)
#         else:
#             print("Invalid method.")
#             return exp_acc
        
#         print(f"Selected samples: {len(train_data)}")
#         print(f"Used indices: {len(used_indices)}")
    
#         initial_train_set = data.ConcatDataset([initial_train_set, train_data])
        
#         # Update remainder by excluding used indices
#         used_indices_set = set(used_indices)
#         remainder_indices = [i for i in range(len(remainder_set)) if i not in used_indices_set]
#         remainder_set = data.Subset(remainder_set, remainder_indices)
        
#         print(f"\nIteration {iteration + 1}")
#         print(f"Train Size: {len(initial_train_set)}, Remainder Size: {len(remainder_set)}")
#         train_loader = data.DataLoader(initial_train_set, batch_size=batch_size, shuffle=True)
#         train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate)

#         test_loader = data.DataLoader(test_set, batch_size=batch_size)
#         accuracy = test_model(model, test_loader)
#         exp_acc.append(accuracy)
#         print(f"Iteration {iteration+1} Accuracy: {accuracy}")

#     return exp_acc

def train_until_empty(model, initial_train_set, remainder_set, test_set,
                      epochs=50, max_iterations=20, batch_size=32,
                      learning_rate=0.01, method=1, run_tag='lcd'):
    
    import numpy as np
    from torch.utils import data
    import csv
    import os

    exp_acc = []

    original_dataset = remainder_set.dataset
    total_data_size = len(original_dataset)

  
    if hasattr(initial_train_set, 'indices'):
        initial_indices = list(initial_train_set.indices)
    else:
        raise ValueError("initial_train_set must be a Subset with indices.")

    # remainder indices
    if hasattr(remainder_set, 'indices'):
        all_remainder_indices = list(remainder_set.indices)
    else:
        all_remainder_indices = list(range(len(original_dataset)))
        all_remainder_indices = [i for i in all_remainder_indices if i not in initial_indices]

   
    available_mask = np.zeros(len(original_dataset), dtype=bool)
    available_mask[all_remainder_indices] = True

    fixed_sample_size = int(0.05 * total_data_size)


    iter_csv = f"time_iter_log_{run_tag}.csv"
    epoch_csv = f"time_epoch_log_{run_tag}.csv"
    if not os.path.exists(iter_csv):
        with open(iter_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method','iteration','train_size_new_samples','remainder_size_after_selection','iteration_time_s','avg_epoch_time_s','total_epoch_time_s','avg_epoch_loss','test_acc'])
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method','iteration','epoch','epoch_time_s','epoch_loss','trained_on']) # trained_on indicates 'initial' or 'new'

    # ----------------- INITIAL TRAINING ON initial_train_set -----------------
    print(">>> Initial training on initial labeled set only")
    if len(initial_indices) > 0:
        init_loader = data.DataLoader(initial_train_set, batch_size=batch_size, shuffle=True)
        init_epoch_losses, init_epoch_times = train_model(model, init_loader, epochs=epochs, learning_rate=learning_rate)
        test_loader = data.DataLoader(test_set, batch_size=batch_size)
        initial_acc = test_model(model, test_loader)
        print(f"Initial training done. Initial test accuracy: {initial_acc:.6f}")
        with open(iter_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([f"method_{method}", 0, len(initial_indices), (available_mask.sum()),  sum(init_epoch_times)+0.0, np.mean(init_epoch_times) if len(init_epoch_times)>0 else 0.0, sum(init_epoch_times), (sum(init_epoch_losses)/len(init_epoch_losses) if len(init_epoch_losses)>0 else 0.0), f"{initial_acc:.6f}"])
        with open(epoch_csv, 'a', newline='') as f:
            w = csv.writer(f)
            for ei, (et, el) in enumerate(zip(init_epoch_times, init_epoch_losses), start=1):
                w.writerow([f"method_{method}", 0, ei, f"{et:.4f}", f"{el:.6f}", "initial"])
    else:
        print("No initial labeled indices provided; skipping initial training.")

    # ----------------- training only on NEW samples each iteration -----------------
    for iteration in range(1, max_iterations+1):
        current_remainder_indices = np.where(available_mask)[0]
        if len(current_remainder_indices) == 0:
            print("Dataset empty. Stopping.")
            break

        current_remainder = data.Subset(original_dataset, current_remainder_indices)
        print(f"\nStarting Iteration {iteration}")
        print(f"Remainder Size (before selection): {len(current_remainder)}")

        # determine selection
        if len(current_remainder) <= fixed_sample_size:
            print("Less than fixed sample size left. Taking all remaining samples.")
            relative_indices = list(range(len(current_remainder)))
            absolute_selected = [current_remainder_indices[i] for i in relative_indices]
            samples_to_add = data.Subset(original_dataset, absolute_selected)
        else:
            if method == 1:
                train_data_subset, relative_indices = LC_HC(model, current_remainder, n=fixed_sample_size)
            elif method == 2:
                train_data_subset, relative_indices = LC_HC_diverse(model, current_remainder, n=fixed_sample_size)
            elif method == 3:
                train_data_subset, relative_indices = HC_diverse(model, current_remainder, n=fixed_sample_size)
            elif method == 4:
                train_data_subset, relative_indices = LC_diverse(model, current_remainder, n=fixed_sample_size)
            else:
                print("Invalid method.")
                return exp_acc

            absolute_selected = [current_remainder_indices[i] for i in relative_indices]
            samples_to_add = data.Subset(original_dataset, absolute_selected)

       
        available_mask[absolute_selected] = False

       
        new_train_size = len(absolute_selected)
        remainder_after = int(available_mask.sum())

        print(f"New samples selected this iteration: {new_train_size}")
        print(f"Remainder Size (after selection): {remainder_after}")

        # ------------------ TIMING: train ONLY on samples_to_add ------------------------
        iter_start = time.time()

        if new_train_size == 0:
            print("No new samples selected this iteration, skipping training.")
            test_loader = data.DataLoader(test_set, batch_size=batch_size)
            acc = test_model(model, test_loader)
            exp_acc.append(acc)
            with open(iter_csv, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([f"method_{method}", iteration, new_train_size, remainder_after, 0.0, 0.0, 0.0, 0.0, f"{acc:.6f}"])
            continue

        train_loader_new = data.DataLoader(samples_to_add, batch_size=batch_size, shuffle=True)

        epoch_losses, epoch_times = train_model(model, train_loader_new, epochs=epochs, learning_rate=learning_rate)

        # test
        test_loader = data.DataLoader(test_set, batch_size=batch_size)
        accuracy = test_model(model, test_loader)

        iter_end = time.time()

        iteration_time = iter_end - iter_start
        total_epoch_time = sum(epoch_times)
        avg_epoch_time = total_epoch_time / len(epoch_times) if len(epoch_times)>0 else 0.0
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if len(epoch_losses)>0 else 0.0

        # Write iteration-level row
        with open(iter_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([f"method_{method}", iteration, new_train_size, remainder_after, f"{iteration_time:.4f}", f"{avg_epoch_time:.4f}", f"{total_epoch_time:.4f}", f"{avg_epoch_loss:.6f}", f"{accuracy:.6f}"])

        # Write epoch-level rows (trained_on='new'):
        with open(epoch_csv, 'a', newline='') as f:
            w = csv.writer(f)
            for ei, (et, el) in enumerate(zip(epoch_times, epoch_losses), start=1):
                w.writerow([f"method_{method}", iteration, ei, f"{et:.4f}", f"{el:.6f}", "new"])

        exp_acc.append(accuracy)
        print(f"Iteration {iteration} Accuracy: {accuracy}")

    return exp_acc
                          
def run_all_methods(model, initial_train_set, remainder, test_set):
    methods = [1, 2, 3, 4]
    results = {}

    # Save initial model state
    initial_model_state = copy.deepcopy(model.state_dict())

    for method in methods:
        print(f"\nStarting training with method {method}")
        
        # Reset model to initial state
        model.load_state_dict(initial_model_state)
        
        # Create deep copies of datasets
        initial_train_set_copy = copy.deepcopy(initial_train_set)
        remainder_copy = copy.deepcopy(remainder)
        
        # Initial training
        train_loader = data.DataLoader(initial_train_set_copy, batch_size=32, shuffle=True)
        train_model(model, train_loader, epochs=1, learning_rate=0.01)
        
        # Initial testing
        test_loader = data.DataLoader(test_set, batch_size=64)
        initial_accuracy = test_model(model, test_loader)
        print(f"Initial accuracy for method {method}: {initial_accuracy}")
        
        # Run the active learning iterations
        exp_acc = train_until_empty(model, initial_train_set_copy, remainder_copy, test_set, 
                                  max_iterations=20, batch_size=32, learning_rate=0.01, method=method)
        results[f"method_{method}"] = exp_acc

    return results
