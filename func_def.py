# func_def.py (finalized)
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
import time
import csv
import os

# Try to import coMA+IB selector (user should have coma_ib/run_coma.py or similar)
try:
    from coma_ib.run_coma import select_topk as coma_select_topk
except Exception:
    coma_select_topk = None  # fallback to existing selectors

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EWC(object):
    def __init__(self, model, dataset, device='cpu', batch_size=64, samples=None, eps=1e-8, clamp_max=1e6):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples = samples
        self.eps = eps
        self.clamp_max = clamp_max

        self._params = {n: p.clone().detach().to(self.device) for n, p in model.named_parameters() if p.requires_grad}
        self._precision_matrices = {n: torch.zeros_like(p).to(self.device) for n, p in model.named_parameters() if p.requires_grad}
        self.compute_fisher()

    def compute_fisher(self):
        self.model.eval()
        loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        count = 0

        for batch_idx, batch in enumerate(loader):
            if self.samples is not None and batch_idx >= self.samples:
                break

            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets, reduction='mean')
            loss.backward()

            # skip batches with invalid grads
            valid = True
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    g = p.grad.data
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        valid = False
                        break
            if not valid:
                self.model.zero_grad()
                continue

            bs = float(inputs.size(0))
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self._precision_matrices[n] += (p.grad.data.clone().pow(2) / bs)

            count += 1
            self.model.zero_grad()

        if count > 0:
            for n in self._precision_matrices:
                mat = self._precision_matrices[n] / float(count)
                mat = torch.nan_to_num(mat, nan=0.0, posinf=self.clamp_max, neginf=0.0)
                mat = mat + self.eps
                mat = torch.clamp(mat, max=self.clamp_max)
                self._precision_matrices[n] = mat

    def penalty(self, model):
        loss = torch.tensor(0., device=self.device)
        for n, p in model.named_parameters():
            if p.requires_grad:
                precision = self._precision_matrices.get(n, None)
                old_param = self._params.get(n, None)
                if precision is None or old_param is None:
                    continue
                diff = (p - old_param).pow(2)
                _loss = precision * diff
                loss = loss + _loss.sum()
        return 0.5 * loss


def prepare_data(dataset_name):
    print('==> Preparing data..')
    if dataset_name == 'cifar10':
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
        raise ValueError("Invalid dataset name. Choose from 'cifar10','cifar100','svhn','voc2012'.")

    torch.manual_seed(seed)
    initial_size = int(0.04 * len(train_set))  # initial 4%
    remainder_size = len(train_set) - initial_size
    initial_train_set, remainder = torch.utils.data.random_split(train_set, [initial_size, remainder_size])

    print(f"Size of initial_train_set: {len(initial_train_set)}")
    print(f"Size of remainder: {len(remainder)}")

    return initial_train_set, remainder, test_set


# ----------------- Training / Testing -----------------
def train_model(model, train_loader, epochs, learning_rate, ewc_obj=None, ewc_lambda=0.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epoch_losses = []
    epoch_times = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for batch_data in train_loader:
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if ewc_obj is not None and ewc_lambda > 0.0:
                pen = ewc_obj.penalty(model)
                if torch.isnan(pen) or torch.isinf(pen):
                    pen = torch.tensor(0., device=device)
                loss = loss + (ewc_lambda * pen)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += float(loss.detach().cpu().item())

        t1 = time.time()
        epoch_time = t1 - t0
        avg_loss = running_loss / max(1, len(train_loader))

        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}, Epoch Time: {epoch_time:.2f}s")

        scheduler.step()

    return epoch_losses, epoch_times


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch in test_loader:
            inputs, labels = data_batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy


def calculate_cluster_centers(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    _ = kmeans.fit_predict(embeddings)
    return kmeans.cluster_centers_


def get_most_diverse_samples(tsne_results, cluster_centers, num_diverse_samples):
    distances = euclidean_distances(tsne_results, cluster_centers)
    min_distances = np.max(distances, axis=1)
    sorted_indices = np.argsort(min_distances)
    diverse_indices = sorted_indices[:num_diverse_samples]
    return diverse_indices


def extract_embeddings(model, dataset):
    test_loader = data.DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            feats = model(images)
            embeddings.extend(feats.view(feats.size(0), -1).cpu().tolist())
    return embeddings


def least_confidence_images(model, test_dataset, k=None):
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    confidences = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
    confidences = torch.tensor(confidences)
    k = min(k, len(confidences)) if k is not None else len(confidences)
    if k == 0:
        return data.Subset(test_dataset, []), []
    _, indices = torch.topk(confidences, k, largest=False)
    return data.Subset(test_dataset, indices), indices.tolist()


def high_confidence_images(model, test_dataset, k=None):
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    confidences = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
    confidences = torch.tensor(confidences)
    k = min(k, len(confidences)) if k is not None else len(confidences)
    if k == 0:
        return data.Subset(test_dataset, []), []
    _, indices = torch.topk(confidences, k, largest=True)
    return data.Subset(test_dataset, indices), indices.tolist()


def HC_diverse(embed_model, remainder, n=None):
    high_conf_images, high_conf_indices = high_confidence_images(embed_model, remainder, k=min(2 * n, len(remainder)) if n else len(remainder))
    high_conf_embeddings = extract_embeddings(embed_model, high_conf_images)
    high_conf_embeddings = np.array([np.array(e) for e in high_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(high_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    diverse_images = data.Subset(high_conf_images, diverse_indices)
    return diverse_images, [high_conf_indices[i] for i in diverse_indices]


def LC_diverse(embed_model, remainder, n=None):
    least_conf_images, least_conf_indices = least_confidence_images(embed_model, remainder, k=min(2 * n, len(remainder)) if n else len(remainder))
    least_conf_embeddings = extract_embeddings(embed_model, least_conf_images)
    least_conf_embeddings = np.array([np.array(e) for e in least_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(least_conf_embeddings)
    cluster_centers = calculate_cluster_centers(tsne_results, 10)
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n)
    diverse_images = data.Subset(least_conf_images, diverse_indices)
    return diverse_images, [least_conf_indices[i] for i in diverse_indices]


def LC_HC(model, remainder, n=None):
    least_confident, least_confident_indices = least_confidence_images(model, remainder, k=(n // 2) if n else 0)
    most_confident, most_confident_indices = high_confidence_images(model, remainder, k=(n // 2) if n else 0)
    combined_dataset = data.ConcatDataset([least_confident, most_confident])
    combined_indices = list(least_confident_indices) + list(most_confident_indices)
    return combined_dataset, combined_indices


def LC_HC_diverse(embed_model, remainder, n, low_conf_ratio=0.5, high_conf_ratio=0.5):
    assert abs(low_conf_ratio + high_conf_ratio - 1.0) < 1e-9, "Ratios must sum to 1.0"

    n_low = int(n * low_conf_ratio)
    n_high = int(n * high_conf_ratio)

    least_conf_images, least_conf_indices = least_confidence_images(embed_model, remainder, k=min(2 * n_low, len(remainder)))
    least_conf_embeddings = extract_embeddings(embed_model, least_conf_images)
    least_conf_embeddings = np.array([np.array(e) for e in least_conf_embeddings])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results_low = tsne.fit_transform(least_conf_embeddings)
    cluster_centers_low = calculate_cluster_centers(tsne_results_low, 10)
    diverse_low_indices = get_most_diverse_samples(tsne_results_low, cluster_centers_low, n_low)
    diverse_least_conf_images = data.Subset(least_conf_images, diverse_low_indices)

    high_conf_images, high_conf_indices = high_confidence_images(embed_model, remainder, k=min(2 * n_high, len(remainder)))
    high_conf_embeddings = extract_embeddings(embed_model, high_conf_images)
    high_conf_embeddings = np.array([np.array(e) for e in high_conf_embeddings])
    tsne_results_high = tsne.fit_transform(high_conf_embeddings)
    cluster_centers_high = calculate_cluster_centers(tsne_results_high, 10)
    diverse_high_indices = get_most_diverse_samples(tsne_results_high, cluster_centers_high, n_high)
    diverse_high_conf_images = data.Subset(high_conf_images, diverse_high_indices)

    combined_dataset = data.ConcatDataset([diverse_least_conf_images, diverse_high_conf_images])
    combined_indices = [least_conf_indices[i] for i in diverse_low_indices] + [high_conf_indices[i] for i in diverse_high_indices]

    return combined_dataset, combined_indices


def train_until_empty(model, initial_train_set, remainder_set, test_set,
                      epochs=50, max_iterations=20, batch_size=32,
                      learning_rate=0.01, method=5, run_tag='ewc_vgg16_cifar10',
                      ewc_lambda=1000.0, lambda_ewc=None, ewc_samples_for_fisher=50):
    if lambda_ewc is not None:
        ewc_lambda = lambda_ewc

    import numpy as np
    from torch.utils import data

    exp_acc = []
    original_dataset = remainder_set.dataset
    total_data_size = len(original_dataset)

    if hasattr(initial_train_set, 'indices'):
        all_selected_indices = set(initial_train_set.indices)
    else:
        raise ValueError("initial_train_set must be a Subset with indices.")

    available_mask = np.ones(len(original_dataset), dtype=bool)
    available_mask[list(all_selected_indices)] = False

    fixed_sample_size = int(0.05 * total_data_size)

    iter_csv = f"time_iter_log_{run_tag}.csv"
    epoch_csv = f"time_epoch_log_{run_tag}.csv"

    if not os.path.exists(iter_csv):
        with open(iter_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'iteration', 'train_size', 'remainder_size',
                        'iteration_time_s', 'avg_epoch_time_s', 'total_epoch_time_s',
                        'avg_epoch_loss', 'test_acc'])
    if not os.path.exists(epoch_csv):
        with open(epoch_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'iteration', 'epoch', 'epoch_time_s', 'epoch_loss'])

    # --- Initial training on initial_train_set only (4%) ---
    print(f"\n=== Initial Training (Iteration 0) ===")
    print(f"Training on initial 4% data: {len(initial_train_set)} samples")
    iter_start = time.time()
    train_loader = data.DataLoader(initial_train_set, batch_size=batch_size, shuffle=True)

    if method == 5:
        consolidated_dataset = data.Subset(original_dataset, list(all_selected_indices))
        ewc_obj = EWC(model, consolidated_dataset, device=device, batch_size=64, samples=ewc_samples_for_fisher)
        epoch_losses, epoch_times = train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate,
                                               ewc_obj=ewc_obj, ewc_lambda=ewc_lambda)
    else:
        epoch_losses, epoch_times = train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate,
                                               ewc_obj=None, ewc_lambda=0.0)

    test_loader = data.DataLoader(test_set, batch_size=batch_size)
    accuracy = test_model(model, test_loader)
    iter_end = time.time()

    total_epoch_time = sum(epoch_times) if len(epoch_times) > 0 else 0.0
    avg_epoch_time = (total_epoch_time / len(epoch_times)) if len(epoch_times) > 0 else 0.0
    avg_epoch_loss = (sum(epoch_losses) / len(epoch_losses)) if len(epoch_losses) > 0 else 0.0

    with open(iter_csv, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([f"method_{method}", 0, len(initial_train_set), len(remainder_set),
                    f"{(iter_end - iter_start):.4f}", f"{avg_epoch_time:.4f}", f"{total_epoch_time:.4f}",
                    f"{avg_epoch_loss:.6f}", f"{accuracy:.6f}"])

    with open(epoch_csv, 'a', newline='') as f:
        w = csv.writer(f)
        for i, (et, el) in enumerate(zip(epoch_times, epoch_losses)):
            w.writerow([f"method_{method}", 0, i + 1, f"{et:.4f}", f"{el:.6f}"])

    exp_acc.append(accuracy)
    print(f"Initial Iteration Accuracy: {accuracy}")

    # --- Iterative loop: add only NEW 5% each iteration ---
    for iteration in range(1, max_iterations + 1):
        current_remainder_indices = np.where(available_mask)[0]
        if len(current_remainder_indices) == 0:
            print("Dataset empty. Stopping.")
            break

        current_remainder = data.Subset(original_dataset, current_remainder_indices)
        print(f"\n=== Iteration {iteration} ===")
        print(f"Remainder Size: {len(current_remainder)}")

        # selection step
        if len(current_remainder) <= fixed_sample_size:
            # take all remaining
            absolute_indices = [int(idx) for idx in current_remainder_indices]
            new_samples = data.Subset(original_dataset, absolute_indices)
        else:
            if method == 1:
                sel_ds, rel_idx = LC_HC(model, current_remainder, n=fixed_sample_size)
                relative_indices = rel_idx
            elif method == 2:
                sel_ds, rel_idx = LC_HC_diverse(model, current_remainder, n=fixed_sample_size)
                relative_indices = rel_idx
            elif method == 3:
                sel_ds, rel_idx = HC_diverse(model, current_remainder, n=fixed_sample_size)
                relative_indices = rel_idx
            elif method == 4:
                sel_ds, rel_idx = LC_diverse(model, current_remainder, n=fixed_sample_size)
                relative_indices = rel_idx
            elif method == 5:
                sel_ds, rel_idx = LC_HC(model, current_remainder, n=fixed_sample_size)
                relative_indices = rel_idx
            elif method == 6:
                # coMA+IB selector (expects relative indices within current_remainder)
                if coma_select_topk is not None:
                    rel_selected = coma_select_topk(model, current_remainder, k=fixed_sample_size, device=device, batch_size=64)
                    relative_indices = [int(x) for x in rel_selected]
                else:
                    # fallback to LC_HC if coMA not available
                    sel_ds, rel_idx = LC_HC(model, current_remainder, n=fixed_sample_size)
                    relative_indices = rel_idx
            else:
                print("Invalid method.")
                return exp_acc

            absolute_indices = [int(current_remainder_indices[i]) for i in relative_indices]
            new_samples = data.Subset(original_dataset, absolute_indices)

        # update masks and seen set
        available_mask[absolute_indices] = False
        all_selected_indices.update(absolute_indices)

        print(f"New samples selected: {len(absolute_indices)}")
        print(f"Total selected samples so far: {len(all_selected_indices)}")

        # For EWC: compute Fisher on all seen data
        ewc_obj = None
        if method == 5:
            consolidated_dataset = data.Subset(original_dataset, list(all_selected_indices))
            ewc_obj = EWC(model, consolidated_dataset, device=device, batch_size=64, samples=ewc_samples_for_fisher)

        # train only on new_samples
        iter_start = time.time()
        train_loader = data.DataLoader(new_samples, batch_size=batch_size, shuffle=True)

        if method == 5:
            epoch_losses, epoch_times = train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate,
                                                   ewc_obj=ewc_obj, ewc_lambda=ewc_lambda)
        else:
            epoch_losses, epoch_times = train_model(model, train_loader, epochs=epochs, learning_rate=learning_rate,
                                                   ewc_obj=None, ewc_lambda=0.0)

        # evaluate
        test_loader = data.DataLoader(test_set, batch_size=batch_size)
        accuracy = test_model(model, test_loader)
        iter_end = time.time()

        total_epoch_time = sum(epoch_times) if len(epoch_times) > 0 else 0.0
        avg_epoch_time = (total_epoch_time / len(epoch_times)) if len(epoch_times) > 0 else 0.0
        avg_epoch_loss = (sum(epoch_losses) / len(epoch_losses)) if len(epoch_losses) > 0 else 0.0

        # logging
        with open(iter_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([f"method_{method}", iteration, len(new_samples),
                        max(0, len(current_remainder) - len(absolute_indices)),
                        f"{(iter_end - iter_start):.4f}", f"{avg_epoch_time:.4f}", f"{total_epoch_time:.4f}",
                        f"{avg_epoch_loss:.6f}", f"{accuracy:.6f}"])

        with open(epoch_csv, 'a', newline='') as f:
            w = csv.writer(f)
            for i, (et, el) in enumerate(zip(epoch_times, epoch_losses)):
                w.writerow([f"method_{method}", iteration, i + 1, f"{et:.4f}", f"{el:.6f}"])

        exp_acc.append(accuracy)
        print(f"Iteration {iteration} Accuracy: {accuracy}")

    return exp_acc


def run_all_methods(model, initial_train_set, remainder, test_set):
    methods = [1, 2, 3, 4, 5, 6]
    results = {}
    initial_model_state = copy.deepcopy(model.state_dict())

    for method in methods:
        print(f"\nStarting training with method {method}")
        model.load_state_dict(initial_model_state)
        initial_train_set_copy = copy.deepcopy(initial_train_set)
        remainder_copy = copy.deepcopy(remainder)

        train_loader = data.DataLoader(initial_train_set_copy, batch_size=32, shuffle=True)
        train_model(model, train_loader, epochs=1, learning_rate=0.01)

        test_loader = data.DataLoader(test_set, batch_size=64)
        initial_accuracy = test_model(model, test_loader)
        print(f"Initial accuracy for method {method}: {initial_accuracy}")

        exp_acc = train_until_empty(model, initial_train_set_copy, remainder_copy, test_set,
                                    max_iterations=20, batch_size=32, learning_rate=0.01, method=method)
        results[f"method_{method}"] = exp_acc

    return results