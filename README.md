# Continual Active Learning Experiments

## Problem Statement

--
## Experiments Performed

We will conducted experiments using the following models and methods:

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
   - Then we use one of the methods to select the next 5% samples to train the model for each iteration
   - At each iteration, only newly pooled samples are used for training, and the previous iteration samples are not used.
   - We retrain the model with these new samples until there is a limit degradation and then check the accuracy after each iteration during retraining for each of these methods.

4. **Parameters:**
   - Learning Rate: 0.01
   - Epochs: 50 for both initial training and sampling iterations
   - Loss Function: Cross Entropy Loss
   - Optimizer: Stochastic Gradient Descent (SGD)
   - Scheduler: StepLR

## Results
--


## Conclusion

--


## Getting Started

To get started, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/vipularya135/Active-Learning-Sampling-Techniques
    cd DAL_Research
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

