import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.detection.ssd import SSDClassificationHead
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import copy
import os
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.vgg import VGG16_Weights
import xml.etree.ElementTree as ET
from torchmetrics.detection import MeanAveragePrecision
import torch.nn.functional as F
from PIL import Image
import random
import warnings
import json
import time
import gc

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VOC Classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')

        # Path to Main/*.txt
        image_sets_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        image_set_file = os.path.join(image_sets_dir, image_set + '.txt')

        if not os.path.isfile(image_set_file):
            raise FileNotFoundError(f"Image set file not found: {image_set_file}")

        with open(image_set_file, 'r') as f:
            image_names = [line.strip().split()[0] for line in f.readlines()]

        self.images = []
        self.annotations = []

        for name in image_names:
            img_path = os.path.join(self.image_dir, name + '.jpg')
            xml_path = os.path.join(self.annotation_dir, name + '.xml')

            if not os.path.exists(xml_path):
                print(f"Warning: Annotation file {xml_path} not found. Skipping image {img_path}")
                continue

            self.images.append(img_path)
            self.annotations.append(xml_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        xml_path = self.annotations[idx]

        image = Image.open(img_path).convert('RGB')

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            return self.create_dummy_annotation(image, idx)

        size = root.find('size')
        width, height = (int(size.find('width').text), int(size.find('height').text)) if size is not None else image.size

        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            if bbox is None:
                continue

            xmin = max(0, float(bbox.find('xmin').text))
            ymin = max(0, float(bbox.find('ymin').text))
            xmax = min(width, float(bbox.find('xmax').text))
            ymax = min(height, float(bbox.find('ymax').text))

            if xmax <= xmin or ymax <= ymin:
                continue

            objects.append({
                'name': name,
                'bbox': [xmin, ymin, xmax, ymax]
            })

        if len(objects) == 0:
            return self.create_dummy_annotation(image, idx)

        boxes = [obj['bbox'] for obj in objects]
        labels = [VOC_CLASSES.index(obj['name']) for obj in objects]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def create_dummy_annotation(self, image, idx):
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((0,), dtype=torch.int64)
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

def detection_transform(image, target):
    """Transform for detection tasks"""
    # Apply transformations to image
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform to image
    orig_width, orig_height = image.size
    image = transform(image)
    
    # Scale bounding boxes to new image size
    if 'boxes' in target and len(target['boxes']) > 0:
        scale_x = 300 / orig_width
        scale_y = 300 / orig_height
        
        boxes = target['boxes'].clone()
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        
        # Clamp boxes to [0, 300]
        boxes[:, 0] = boxes[:, 0].clamp(0, 300)
        boxes[:, 1] = boxes[:, 1].clamp(0, 300)
        boxes[:, 2] = boxes[:, 2].clamp(0, 300)
        boxes[:, 3] = boxes[:, 3].clamp(0, 300)
        
        # Remove boxes with invalid dimensions
        valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_indices]
        labels = target['labels'][valid_indices]
        
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    
    return image, target

def prepare_data():
    print('==> Preparing VOC2012 data...')

    # Corrected dataset path
    train_val_path = "data/VOC2012/VOCdevkit/VOC2012"

    # Load PascalVOC datasets
    train_set = PascalVOCDataset(
        root_dir=train_val_path,
        image_set='trainval',
        transform=detection_transform
    )

    test_set = PascalVOCDataset(
        root_dir=train_val_path,
        image_set='val',
        transform=detection_transform
    )

    # Filter out invalid data
    train_filtered = [item for item in train_set if item is not None]
    test_filtered = [item for item in test_set if item is not None]

    # Convert filtered lists to dataset
    class ListDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

    train_dataset = ListDataset(train_filtered)
    test_dataset = ListDataset(test_filtered)

    # Split into 4% initial training set and 96% remainder
    torch.manual_seed(42)
    initial_size = int(0.04 * len(train_dataset))
    remainder_size = len(train_dataset) - initial_size

    initial_train_set, remainder = random_split(train_dataset, [initial_size, remainder_size])

    print(f"Initial train set size: {len(initial_train_set)}")
    print(f"Remainder set size: {len(remainder)}")
    print(f"Test set size: {len(test_dataset)}")

    return initial_train_set, remainder, test_dataset

# Create model with COCO pre-trained backbone
def create_model(num_classes=21):  # 20 classes + background
    try:
        # Load SSD300 with VGG16 backbone pre-trained on COCO
        model = ssd300_vgg16(
            weights='COCO_V1',  # Load COCO pre-trained weights
            num_classes=num_classes
        )
        print("Using SSD300 with VGG16 pretrained on MS COCO backbone")
    except Exception as e:
        print(f"COCO pretrained weights unavailable: {e}")
        # Fall back to ImageNet if COCO weights not available
        try:
            model = ssd300_vgg16(
                weights=None,
                weights_backbone=VGG16_Weights.IMAGENET1K_V1,
                num_classes=num_classes
            )
            print("Falling back to ImageNet pretrained backbone")
        except Exception as e:
            print(f"ImageNet pretrained weights unavailable: {e}")
            model = ssd300_vgg16(weights=None, num_classes=num_classes)
            print("Using randomly initialized backbone")

    return model.to(device)


# Collate function for detection that filters None values
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return [], []
    
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# Enhanced training function with better hyperparameters
def train_model(model, train_loader, epochs, learning_rate):
    # Freeze backbone layers initially
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
    
    # Remove 'verbose' parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    warmup_epochs = min(3, epochs//2)
    warmup_factor = 0.1
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * warmup_factor * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        for images, targets in train_loader:
            if len(images) == 0:
                continue
                
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += losses.item()
            batch_count += 1
        
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
            
            # Manual LR change logging
            old_lr = optimizer.param_groups[0]['lr']
            if epoch >= warmup_epochs:
                scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if new_lr != old_lr:
                print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
        else:
            print(f'Epoch {epoch+1}/{epochs}: No valid batches')
    
    if epochs > 5:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = True
        print("Unfrozen backbone layers")

# Enhanced evaluation with lower confidence threshold
def test_model(model, test_loader):
    map_metric = MeanAveragePrecision(
        iou_type="bbox", 
        class_metrics=False,
        iou_thresholds=[0.25, 0.5, 0.75]
    )
    model.eval()
    
    with torch.no_grad():
        for images, targets in test_loader:
            if len(images) == 0:
                continue
                
            images = list(image.to(device) for image in images)
            predictions = model(images)
            
            formatted_preds = []
            formatted_targets = []
            
            for i in range(len(images)):
                pred = predictions[i]
                targ = targets[i]
                
                keep = pred['scores'] > 0.005
                filtered_boxes = pred['boxes'][keep]
                filtered_scores = pred['scores'][keep]
                filtered_labels = pred['labels'][keep]
                
                formatted_preds.append({
                    'boxes': filtered_boxes.cpu(),
                    'scores': filtered_scores.cpu(),
                    'labels': filtered_labels.cpu()
                })
                
                formatted_targets.append({
                    'boxes': targ['boxes'].cpu(),
                    'labels': targ['labels'].cpu()
                })
            
            map_metric.update(formatted_preds, formatted_targets)
    
    results = map_metric.compute()
    
    # Handle different key formats for IoU thresholds
    def get_iou_key(base, threshold):
        # Try integer key (e.g., 'map_25')
        int_key = f"{base}_{int(threshold*100)}"
        if int_key in results:
            return results[int_key].item()
        # Try float key (e.g., 'map_0.25')
        float_key = f"{base}_{threshold}"
        if float_key in results:
            return results[float_key].item()
        return 0.0  # Default if key not found

    map_score = results['map'].item()
    iou_50 = get_iou_key('map', 0.50)
    
    print(f"Test mAP: {map_score:.4f}")
    print(f"Test IoU: {iou_50:.4f}")
    
    return {
        'map': map_score,
        'iou': iou_50,
    }

# Confidence calculation for detection
def calculate_image_confidence(outputs):
    """Calculate confidence score for an image based on its detections"""
    if len(outputs['scores']) == 0:
        return torch.tensor(0.0)  # No detections
    return outputs['scores'].max()

def least_confidence_images(model, dataset, k=None):
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), collate_fn=collate_fn, shuffle=False)
    confidences = []
    valid_indices = []
    global_idx = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if len(images) == 0:
                continue
                
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                conf = calculate_image_confidence(output).item()
                confidences.append(conf)
                valid_indices.append(global_idx)
                global_idx += 1
    
    if len(confidences) == 0:
        return Subset(dataset, []), []
        
    confidences = torch.tensor(confidences)
    k = min(k, len(confidences)) if k is not None else len(confidences)
    _, indices = torch.topk(confidences, k, largest=False)
    actual_indices = [valid_indices[i] for i in indices]
    return Subset(dataset, actual_indices), actual_indices

def high_confidence_images(model, dataset, k=None):
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), collate_fn=collate_fn, shuffle=False)
    confidences = []
    valid_indices = []
    global_idx = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if len(images) == 0:
                continue
                
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                conf = calculate_image_confidence(output).item()
                confidences.append(conf)
                valid_indices.append(global_idx)
                global_idx += 1
    
    if len(confidences) == 0:
        return Subset(dataset, []), []
        
    confidences = torch.tensor(confidences)
    k = min(k, len(confidences)) if k is not None else len(confidences)
    _, indices = torch.topk(confidences, k, largest=True)
    actual_indices = [valid_indices[i] for i in indices]
    return Subset(dataset, actual_indices), actual_indices

# Feature extraction with larger batch size
def extract_embeddings(model, dataset):
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), collate_fn=collate_fn, shuffle=False)
    embeddings = []
    valid_indices = []
    
    # Use the backbone to extract features
    backbone = model.backbone
    
    model.eval()
    with torch.no_grad():
        global_idx = 0
        for batch_idx, (images, targets) in enumerate(loader):
            if len(images) == 0:
                continue
                
            # Stack images into a batch tensor [N, C, H, W]
            images_tensor = torch.stack(images, dim=0).to(device)
            features = backbone(images_tensor)
            
            # Extract the first feature map (key '0')
            if isinstance(features, dict):
                if '0' in features:
                    features = features['0']  # Shape: [N, C, H, W]
                else:
                    # Try to get the first key if '0' doesn't exist
                    first_key = next(iter(features.keys()))
                    features = features[first_key]
            
            # Global average pooling
            pooled = F.adaptive_avg_pool2d(features, (1, 1))
            # Flatten to [N, C]
            pooled = pooled.squeeze(-1).squeeze(-1).cpu().numpy()
            
            embeddings.extend(pooled)
            
            # Record global indices for this batch
            for _ in images:
                valid_indices.append(global_idx)
                global_idx += 1
    
    return embeddings, valid_indices

# Diversity methods
def calculate_cluster_centers(embeddings, num_clusters):
    if len(embeddings) < num_clusters:
        num_clusters = max(1, len(embeddings) // 2)
        
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10, max_iter=500)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers

def get_most_diverse_samples(tsne_results, cluster_centers, num_diverse_samples):
    distances = euclidean_distances(tsne_results, cluster_centers)
    min_distances = np.min(distances, axis=1)  # Distance to closest cluster center
    diversity_scores = np.max(distances, axis=1)  # Max distance to any cluster center
    
    # Combine both metrics
    combined_scores = min_distances * diversity_scores
    
    sorted_indices = np.argsort(combined_scores)[::-1]  # Descending order
    diverse_indices = sorted_indices[:num_diverse_samples]
    return diverse_indices

def HC_diverse(model, remainder, n=None):
    high_conf_images, high_conf_indices = high_confidence_images(
        model, remainder, k=min(2*n, len(remainder)) if n else len(remainder)
    )
    if len(high_conf_images) == 0:
        return Subset(remainder, []), []
        
    high_conf_embeddings, valid_indices = extract_embeddings(model, high_conf_images)
    
    if len(high_conf_embeddings) == 0 or len(high_conf_embeddings) <= 1:
        return Subset(remainder, []), []
    
    # Use PCA for dimensionality reduction first if needed
    tsne = TSNE(
        n_components=2, 
        perplexity=min(30, len(high_conf_embeddings)-1), 
        n_iter=500, 
        learning_rate=200
    )
    tsne_results = tsne.fit_transform(high_conf_embeddings)
    
    num_clusters = min(10, len(tsne_results))
    cluster_centers = calculate_cluster_centers(tsne_results, num_clusters)
    n_samples = min(n, len(tsne_results))
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n_samples)
    
    actual_indices = [high_conf_indices[i] for i in diverse_indices]
    diverse_images = Subset(high_conf_images, diverse_indices)
    return diverse_images, actual_indices

def LC_diverse(model, remainder, n=None):
    least_conf_images, least_conf_indices = least_confidence_images(
        model, remainder, k=min(2*n, len(remainder)) if n else len(remainder)
    )
    if len(least_conf_images) == 0:
        return Subset(remainder, []), []
        
    least_conf_embeddings, valid_indices = extract_embeddings(model, least_conf_images)
    
    if len(least_conf_embeddings) == 0 or len(least_conf_embeddings) <= 1:
        return Subset(remainder, []), []
    
    tsne = TSNE(
        n_components=2, 
        perplexity=min(30, len(least_conf_embeddings)-1), 
        n_iter=500, 
        learning_rate=200
    )
    tsne_results = tsne.fit_transform(least_conf_embeddings)
    
    num_clusters = min(10, len(tsne_results))
    cluster_centers = calculate_cluster_centers(tsne_results, num_clusters)
    n_samples = min(n, len(tsne_results))
    diverse_indices = get_most_diverse_samples(tsne_results, cluster_centers, n_samples)
    
    actual_indices = [least_conf_indices[i] for i in diverse_indices]
    diverse_images = Subset(least_conf_images, diverse_indices)
    return diverse_images, actual_indices

def LC_HC(model, remainder, n=None):
    n_samples = min(n, len(remainder))
    n_low = n_samples // 2
    n_high = n_samples - n_low
    
    least_confident, least_confident_indices = least_confidence_images(model, remainder, k=n_low)
    most_confident, most_confident_indices = high_confidence_images(model, remainder, k=n_high)
    
    # Combine datasets and indices
    combined_dataset = torch.utils.data.ConcatDataset([least_confident, most_confident])
    combined_indices = least_confident_indices + most_confident_indices
    
    return combined_dataset, combined_indices

def LC_HC_diverse(model, remainder, n, low_conf_ratio=0.5, high_conf_ratio=0.5):
    n_samples = min(n, len(remainder))
    n_low = int(n_samples * low_conf_ratio)
    n_high = n_samples - n_low

    # Low confidence diverse
    low_conf_dataset, low_conf_indices = LC_diverse(model, remainder, n_low)
    
    # High confidence diverse
    high_conf_dataset, high_conf_indices = HC_diverse(model, remainder, n_high)
    
    # Combine
    combined_dataset = torch.utils.data.ConcatDataset([low_conf_dataset, high_conf_dataset])
    combined_indices = low_conf_indices + high_conf_indices
    
    return combined_dataset, combined_indices

# Active learning loop with better resource management
def train_until_empty(model, initial_train_set, remainder_set, test_set,
                      epochs=1, max_iterations=10, batch_size=8,
                      learning_rate=0.0001, method=1):
    
    # Get the original dataset from the remainder
    if isinstance(remainder_set, Subset):
        original_dataset = remainder_set.dataset
        # Create availability mask for entire dataset
        total_size = len(original_dataset)
        available_mask = np.ones(total_size, dtype=bool)
        # Mark initial training indices as unavailable
        initial_indices = set(initial_train_set.indices)
        for idx in initial_indices:
            if idx < total_size:
                available_mask[idx] = False
    else:
        # Handle other dataset types
        total_size = len(remainder_set)
        available_mask = np.ones(total_size, dtype=bool)
        original_dataset = remainder_set
    
    # Track all selected indices
    all_selected_indices = set(initial_train_set.indices)
    
    # Fixed sample size per iteration (5% of total)
    fixed_sample_size = max(1, int(0.05 * total_size))
    exp_metrics = {
        'map': [],
        'iou_50': [],
    }
    
    for iteration in range(max_iterations):
        # Create current remainder dataset
        current_remainder_indices = np.where(available_mask)[0]
        if len(current_remainder_indices) == 0:
            print("Dataset empty. Stopping.")
            break
            
        current_remainder = Subset(original_dataset, current_remainder_indices)
        print(f"\nIteration {iteration+1}: Remainder size={len(current_remainder)}")
        
        # Select new samples
        if len(current_remainder) <= fixed_sample_size:
            print("Using all remaining samples")
            new_indices = list(range(len(current_remainder)))
            new_samples = current_remainder
        else:
            if method == 1:
                new_samples, new_indices = LC_HC(model, current_remainder, fixed_sample_size)
            elif method == 2:
                new_samples, new_indices = LC_HC_diverse(model, current_remainder, fixed_sample_size)
            elif method == 3:
                new_samples, new_indices = HC_diverse(model, current_remainder, fixed_sample_size)
            elif method == 4:
                new_samples, new_indices = LC_diverse(model, current_remainder, fixed_sample_size)
            else:
                raise ValueError("Invalid method")
        
        # Convert relative indices to absolute indices in original dataset
        absolute_indices = [current_remainder_indices[i] for i in new_indices]
        print(f"Selected {len(absolute_indices)} new samples")
        
        # Update availability mask
        available_mask[absolute_indices] = False
        all_selected_indices.update(absolute_indices)
        
        # Update training set
        initial_train_set = Subset(original_dataset, list(all_selected_indices))
        
        # Train
        train_loader = DataLoader(
            initial_train_set, 
            batch_size=min(batch_size, len(initial_train_set)), 
            shuffle=True, 
            collate_fn=collate_fn
        )
        train_model(model, train_loader, epochs, learning_rate)
        
        # Evaluate
        test_loader = DataLoader(
            test_set, 
            batch_size=min(8, len(test_set)), 
            collate_fn=collate_fn, 
            shuffle=False
        )
        metrics = test_model(model, test_loader)
        
        # Store all metrics
        exp_metrics['map'].append(metrics['map'])
        exp_metrics['iou_50'].append(metrics['iou_50'])
        
        print(f"Iteration {iteration+1} mAP: {metrics['map']:.4f}")
        print(f"iou: {metrics['iou_50']:.4f}")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    
    return exp_metrics

def run_all_methods(initial_train_set, remainder, test_set):
    methods = [1, 2, 3, 4]
    results = {}
    
    for method in methods:
        print(f"\n=== Starting method {method} ===")
        # Create new model for each method
        model = create_model(num_classes=len(VOC_CLASSES)+1)
        
        # Initial training with more epochs
        train_loader = DataLoader(
            initial_train_set, 
            batch_size=min(8, len(initial_train_set)), 
            shuffle=True, 
            collate_fn=collate_fn
        )
        train_model(model, train_loader, epochs=1, learning_rate=0.0001)
        
        # Initial evaluation
        test_loader = DataLoader(
            test_set, 
            batch_size=min(8, len(test_set)), 
            collate_fn=collate_fn, 
            shuffle=False
        )
        initial_metrics = test_model(model, test_loader)
        print(f"Initial metrics for method {method}:")
        print(f"  mAP: {initial_metrics['map']:.4f}")
        print(f"  iou: {initial_metrics['iou_50']:.4f}")
        
        # Run active learning
        exp_metrics = train_until_empty(
            model, initial_train_set, remainder, test_set,
            epochs=1, max_iterations=10, batch_size=8,
            learning_rate=0.0001, method=method
        )
        
        # Store all metrics including initial
        results[f"method_{method}"] = {
            'initial': initial_metrics,
            'iterations': exp_metrics
        }
        
        # Clean up to save memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    return results

def main():
    print(f"Using device: {device}")
    start_time = time.time()
    
    # Prepare data
    initial_train_set, remainder, test_set = prepare_data()
    
    # Run experiments
    results = run_all_methods(initial_train_set, remainder, test_set)
    
    # Save results in JSON format
    with open('detection_results.json', 'w') as f:
        # Convert tensors to floats for JSON serialization
        json_results = {}
        for method, data in results.items():
            json_results[method] = {
                'initial': {k: float(v) for k, v in data['initial'].items()},
                'iterations': {k: [float(v) for v in vals] for k, vals in data['iterations'].items()}
            }
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print("\n=== Final Results ===")
    for method, data in results.items():
        print(f"\nMethod: {method}")
        print("Initial Metrics:")
        print(f"  mAP: {data['initial']['map']:.4f}")
        print(f"  iou: {data['initial']['iou_50']:.4f}")
        
        print("\nIteration Metrics:")
        print("Iter | mAP     |  mAP@0.50 ")
        for i in range(len(data['iterations']['map'])):
            print(f"{i+1:4d} | {data['iterations']['map'][i]:.4f} | {data['iterations']['iou_50'][i]:.4f} ")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()