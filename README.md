# Active Learning Experiments

## Problem Statement

Deep learning models including Convolutional Neural Networks and Vision Transformers have achieved state-of-the-art performance on many computer vision tasks such as object classification, detection, segmentation, generation and many more. However, these models are data hungry since they require large amount of training data to learn the huge number of parameters or weights. Especially working with supervised learning tasks, curating a large number of labeled images for model training is an expensive and time consuming task. Active Learning (AL) has been used to address this problem for many years. Existing active learning methods aim at choosing the samples for annotation from a pool of unlabelled set that are either diverse or uncertain. Choosing such samples may hinder the model performance since we are pooling w.r.t. one dimension i.e., either diverse or uncertain. In this paper, we propose a novel hybrid sampling method for pooling both easy and hard samples which are also diverse. To verify the efficacy of the proposed method, experiments are conducted considering high and low confidence samples separately. It is evident from the experimental results that the proposed hybrid sampling method helps the deep learning models to achieve better results. 

## Experiments Performed

We conducted experiments using the following models and methods:

1. **Models:**
   - VGG-16
   - ResNet-18
   - ResNet-50
   - ResNet-56
   - Mobilenet
   - DenseNet-121
   - ViT-Small Transformer
   - Swin Transformer

2. **Datasets:**
   - CIFAR-10
   - CIFAR-100
   - SVHN
   - PascalVOC-2012
   - Tiny ImageNet

3. **Experiment Details:**
   - We train our model initially on 4% of total data samples.
   - Then we use one of the 4 methods to select the next 5% samples to train the model for each iteration:
     1. Method 1: top 5k samples that are diverse low and diverse high confidence.
     2. Method 2: top 5k samples that are diverse and high confidence.
     3. Method 3: top 5k samples that are diverse and low confidence.
     4. Method 4: top 5k samples that are low and high confidence.
   - We retrain the model with these new samples until there is a limit degradation and then check the accuracy after each iteration during retraining for each of these methods.

4. **Parameters:**
   - Learning Rate: 0.01
   - Epochs: 50 for both initial training and sampling iterations
   - Loss Function: Cross Entropy Loss
   - Optimizer: Stochastic Gradient Descent (SGD)
   - Scheduler: StepLR

## Results
### Performance Comparison of Hybrid Active Learning Methods

| Dataset            | Model          | LC + Diverse (LCD) | HC + Diverse (HCD) | LC + HC (LCHC) | LC + HC + Diverse (DSAL) |
| :----------------- | :------------- | :----------------- | :----------------- | :------------- | :----------------------- |
| **CIFAR-10** | VGG-16         | 93.87%             | 93.02%             | 93.64%         | 93.83%                   |
|                    | ResNet-18      | 95.00%             | 94.55%             | 94.83%         | 94.91%                   |
|                    | ResNet-50      | 95.00%             | 95.27%             | 95.48%         | 94.91%                   |
|                    | ResNet-56      | 95.26%             | 95.11%             | 95.25%         | 95.09%                   |
|                    | Mobilenet      | 91.39%             | 90.16%             | 91.14%         | 90.85%                   |
|                    | DenseNet-121   | 95.35%             | 94.99%             | 95.50%         | 95.30%                   |
|                    | Swin           | 85.40%             | 82.95%             | 83.89%         | 83.61%                   |
|                    | ViT-Small      | 83.05%             | 81.35%             | 82.10%         | 82.33%                   |
| **CIFAR-100** | ResNet-18      | 74.86%             | 74.28%             | 74.58%         | 74.77%                   |
|                    | ResNet-50      | 78.41%             | 78.01%             | 78.28%         | 78.37%                   |
|                    | DenseNet-121   | 78.49%             | 78.01%             | 78.16%         | 77.99%                   |
|                    | Swin           | 65.37%             | 64.72%             | 63.91%         | 64.77%                   |
| **SVHN** | VGG-16         | 95.63%             | 95.49%             | 95.44%         | 95.31%                   |
|                    | ResNet-18      | 95.56%             | 95.41%             | 95.19%         | 95.45%                   |
|                    | ResNet-50      | 95.93%             | 95.94%             | 95.33%         | 95.61%                   |
|                    | ResNet-56      | 94.62%             | 95.61%             | 95.33%         | 95.61%                   |
|                    | Mobilenet      | 94.97%             | 94.91%             | 94.73%         | 94.04%                   |
|                    | DenseNet-121   | 96.38%             | 96.19%             | 96.10%         | 96.09%                   |
| **Tiny ImageNet** | ResNet-18      | 64.87%             | 63.44%             | 64.18%         | 64.15%                   |
|                    | ResNet-50      | 66.57%             | 66.40%             | 66.31%         | 66.46%                   |
| **PascalVOC 2012** | VGG-16         | 84.44%             | 84.37%             | 84.38%         | 84.37%                   |


## Conclusion

Our experiments indicate that selecting low confidence and diverse samples generally results in the highest accuracy improvement across various models and datasets. The order of effectiveness from best to worst in our experiments was:
1. Diverse and low confidence
2. Diverse and low + high confidence
3. Low and high confidence
4. Diverse and high confidence



## Getting Started

To get started, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/vipularya135/Active-Learning-Sampling-Techniques
    cd Active-Learning-Sampling-Techniques
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the `main.py` script, by selecting the model and dataset:
    ```bash
    python main.py 
    ```

## File Descriptions

### 1. Function_Definitions.py
This file contains the definitions of various functions used throughout the project. These functions might include data preprocessing, model evaluation, and utility functions to streamline the workflow.

### 2. main.py
This is the main execution script of the project. It coordinates the entire process, from data loading and preprocessing to model training and evaluation. It serves as the entry point to run the project.

### 3. models.py
This file defines the architecture of the machine learning or deep learning models used in the project. It may include custom model classes, layers, and configurations necessary for training and inference.

### 4. requirements.txt
This file lists all the Python dependencies and libraries required to run the project. It ensures that anyone who clones the repository can install the exact versions of the dependencies needed to avoid compatibility issues.

### 5. swin.py
This file implements the Swin Transformer model, a type of Vision Transformer known for its hierarchical design and efficiency in handling computer vision tasks.

### 6. vit-tiny.py
This file implements the Vision Transformer (ViT) model, specifically the tiny variant, which is designed for image classification tasks with a smaller parameter count and faster training times.
### 7. VOC2012-Expirements.py
This file implements the VGG16 model(Pretrained on MS COCO Dataset) on VOC2012 dataset, which is designed for Object Detection tasks.

