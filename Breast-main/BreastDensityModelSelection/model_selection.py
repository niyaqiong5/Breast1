import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

class ModelRegistry:
    """Model registry to store and retrieve model architectures"""
    def __init__(self):
        self.models = {}
        
    def register(self, name):
        """Register a model architecture"""
        def decorator(model_class):
            self.models[name] = model_class
            return model_class
        return decorator
        
    def get_model(self, name):
        """Get a model architecture by name"""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry. Available models: {list(self.models.keys())}")
        return self.models[name]
    
    def get_all_models(self):
        """Get all registered model architectures"""
        return self.models
    
# Create a global model registry
model_registry = ModelRegistry()

# Improved BEiT model - Focused on medical image adaptation
@model_registry.register("BEiTBreastDensityNet")
class ImprovedBEiTBreastDensityNet(nn.Module):
    '''
Improved BEiT model implementation:

1. Use specially designed channel adaptation blocks to retain more original information
2. Use a miniaturized BEiT model that is easier to adapt to medical image tasks
3. Add a domain adaptation layer to help the model adapt to medical image features
4. Use a class-weighted classification head to improve rare class performance
5. Add residual connections and feature fusion to improve feature utilization
    '''
    def __init__(self, num_classes=4, model_name="beit_base_patch16_224"):
        super(ImprovedBEiTBreastDensityNet, self).__init__()
        
        # 1. Advanced channel adaptation block, retaining more original information
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(20, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.GELU()
        )
        
        # 2. Use smaller BEiT models, which are easier to transfer learning
        self.backbone = timm.create_model(
            model_name,  # 使用beit_small而不是beit_base
            pretrained=True, 
            num_classes=0,
            in_chans=3
        )
        
        if hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            feature_dim = self.backbone.embed_dim
        
        # 3. Domain Adaptation Layer - Adapting to Medical Image Features
        self.domain_adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.attention = AttentionBlock(feature_dim)
        
        # 4. Improved classification head, using deeper feature extraction
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # 5. To maintain stability during fine-tuning, freeze some backbone layers
        for param in list(self.backbone.parameters())[:-20]: 
            param.requires_grad = False
    
    def forward(self, x):
        if x.size(-1) != 224 or x.size(-2) != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.channel_adapter(x)
        
        features = self.backbone(x)
        
        adapted_features = self.domain_adapter(features)
        
        adapted_features = self.attention(adapted_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        features = features + adapted_features
        
        logits = self.classifier(features)
        
        return logits


@model_registry.register("BEiTWithCNNBreastDensityNet")
class BEiTWithCNNBreastDensityNet(nn.Module):
    '''
Hybrid BEiT-CNN architecture:

1. Use CNN front-end for medical image feature extraction
2. Combine BEiT's global feature representation capability
3. Obtain more comprehensive representation through feature fusion
    '''
    def __init__(self, num_classes=4):
        super(BEiTWithCNNBreastDensityNet, self).__init__()
        
        # CNN front end for feature extraction - saving intermediate features
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Generates 3-channel output for BEiT
        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        # BEiT backbone
        self.backbone = timm.create_model(
            "beit_base_patch16_224",  
            pretrained=True, 
            num_classes=0,
            in_chans=3
        )
        
        # Freeze part of the BEiT layer
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        feature_dim = self.backbone.embed_dim
        
        self.cnn_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 64, 512),  # Fusion of CNN and Transformer features
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x1 = self.cnn_layer1(x)
        x2 = self.cnn_layer2(x1)
        
        # Extracting CNN features
        cnn_features = self.cnn_features(x2).view(batch_size, -1)
        
        x_rgb = self.to_rgb(x2)
        
        # Resize to 224x224 for BEiT
        if x_rgb.size(-1) != 224 or x_rgb.size(-2) != 224:
            x_rgb = torch.nn.functional.interpolate(x_rgb, size=(224, 224), mode='bilinear', align_corners=False)
        
        transformer_features = self.backbone(x_rgb)
        
        combined_features = torch.cat([transformer_features, cnn_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        logits = self.classifier(fused_features)
        
        return logits

@model_registry.register("BreastDensity3DNet")
class BreastDensity3DNet(nn.Module):
    '''
BreastDensity3DNet:

This is the basic model architecture, using the traditional CNN structure

Use 4 convolution blocks, each containing a convolution layer, batch normalization, ReLU activation function, and max pooling

Use the attention mechanism (Attention Block) between convolution blocks to enhance feature extraction

Finally, a larger fully connected layer (1024→512→256→4) is used for classification

A decreasing Dropout (0.5→0.4→0.3) is applied in the fully connected layer to prevent overfitting

Suitable for general classification tasks, balancing performance and computational complexity
    '''

    def __init__(self, num_classes=4):
        super(BreastDensity3DNet, self).__init__()
        
        # Input: [batch, channels=20, height=256, width=256]
        # 20 channels: 10 for breast mask slices, 10 for glandular mask slices
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention1 = AttentionBlock(32)
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention2 = AttentionBlock(64)
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention3 = AttentionBlock(128)
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers with improved dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Apply convolutional blocks with attention
        x = self.conv_block1(x)
        x = self.attention1(x)
        
        x = self.conv_block2(x)
        x = self.attention2(x)
        
        x = self.conv_block3(x)
        x = self.attention3(x)
        
        x = self.conv_block4(x)
        
        # Flatten and apply fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x


@model_registry.register("LightweightBreastDensityNet")
class LightweightBreastDensityNet(nn.Module):

    '''
LightweightBreastDensityNet:

Lightweight model with significantly fewer parameters than other models

Only 3 convolution blocks are used, and the number of convolution kernels is small (16→32→64)

Global Average Pooling is used instead of a large number of fully connected layers to significantly reduce the number of parameters

Only a simple hidden layer with 128 nodes

No attention mechanism is used

The advantage is fast training and inference speed, suitable for resource-constrained environments, but the accuracy may be slightly lower on complex data
'''
    def __init__(self, num_classes=4):
        super(LightweightBreastDensityNet, self).__init__()
        
        # Input: [batch, channels=20, height=256, width=256]
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(20, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling instead of flatten + FC layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Small FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x


@model_registry.register("DeepBreastDensityNet")
class DeepBreastDensityNet(nn.Module):

    '''DeepBreastDensityNet：

Deeper network structure, each convolution block contains two consecutive convolution layers

Uses 4 convolution blocks with attention mechanism

Maintains the same fully connected layer structure as the basic model

Increased depth enables the model to have stronger feature extraction and expression capabilities

Suitable for processing more complex features and more subtle classification differences

Disadvantages are larger number of parameters, longer training time, and higher risk of overfitting
'''
    def __init__(self, num_classes=4):
        super(DeepBreastDensityNet, self).__init__()
        
        # Input: [batch, channels=20, height=256, width=256]
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention1 = AttentionBlock(32)
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention2 = AttentionBlock(64)
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention3 = AttentionBlock(128)
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention4 = AttentionBlock(256)
        
        # Fully connected layers with improved dropout
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.attention1(x)
        
        x = self.conv_block2(x)
        x = self.attention2(x)
        
        x = self.conv_block3(x)
        x = self.attention3(x)
        
        x = self.conv_block4(x)
        x = self.attention4(x)
        
        # Flatten and apply fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x


@model_registry.register("ResNetBreastDensityNet")
class ResNetBreastDensityNet(nn.Module):
    
    '''ResNetBreastDensityNet:

Based on the ResNet architecture design, the core is to use residual connections

Contains a special ResBlock module that allows information to directly skip certain layers (shortcut connections)

Uses a larger initial convolution kernel (7x7) and a deeper layered structure

Uses global average pooling at the end of the network to reduce the number of parameters

The biggest advantage is that it can train deeper networks without being affected by the gradient disappearance/explosion problem

Suitable for more complex data sets and scenarios that require higher accuracy
'''
    def __init__(self, num_classes=4):
        super(ResNetBreastDensityNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.attention = AttentionBlock(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        layers.append(ResBlock(in_channels, out_channels, stride))

        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.attention(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))

        return x * attention


# Model Selector Class
class ModelSelector:
    def __init__(self, device, registry=model_registry):
        self.device = device
        self.registry = registry
        self.models = {}
        self.results = {}
        
    def add_model(self, model_name, **model_params):
        # Add a model to the candidate list
        model_class = self.registry.get_model(model_name)
        model = model_class(**model_params).to(self.device)
        self.models[model_name] = model
        return model
    
    def add_all_models(self, **default_params):
        for name, model_class in self.registry.get_all_models().items():
            self.add_model(name, **default_params)
        return self.models
        
    def train_and_evaluate(self, model_name, train_loader, val_loader, criterion, 
                           optimizer_class=torch.optim.Adam, optimizer_params=None,
                           scheduler_class=None, scheduler_params=None,
                           epochs=50, early_stopping_patience=10,
                           use_mixed_precision=True, rare_classes=None):
        # Training and evaluating a specific model
        if optimizer_params is None:
            optimizer_params = {'lr': 0.001}
            
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Add it first with add_model().")
            
        model = self.models[model_name]
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        
        if scheduler_class:
            scheduler = scheduler_class(optimizer, **scheduler_params)
        else:
            scheduler = None
            
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_model_state = None
        no_improve_epochs = 0
        history = {
            'train_loss': [], 
            'val_loss': [], 
            'train_acc': [], 
            'val_acc': [],
            'lr': []
        }
        
        if rare_classes:
            history['rare_class_metrics'] = {cls: {'acc': [], 'recall': [], 'precision': []} for cls in rare_classes}
        
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
        
        # train
        print(f"开始训练模型 {model_name}，共 {epochs} 个epoch")
        start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Tracking rare class
            if rare_classes:
                rare_class_correct = {cls: 0 for cls in rare_classes}
                rare_class_total = {cls: 0 for cls in rare_classes}
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                inputs, labels, is_synthetic = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                
                if use_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Rare class statistics
                if rare_classes:
                    for cls in rare_classes:
                        cls_mask = (labels == cls)
                        if torch.sum(cls_mask) > 0:
                            rare_class_total[cls] += torch.sum(cls_mask).item()
                            rare_class_correct[cls] += torch.sum(predicted[cls_mask] == labels[cls_mask]).item()
            
            # Calculate the average loss and accuracy
            train_loss = train_loss / train_total if train_total > 0 else 0
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            # validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    inputs, labels, _ = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
  
            val_loss = val_loss / val_total if val_total > 0 else float('inf')
            val_acc = val_correct / val_total if val_total > 0 else 0

            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer_params['lr']

            if rare_classes and all_labels:
                conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(4)))
                
                for cls in rare_classes:
                    if cls < len(conf_matrix):
                        # Calculate the average loss and accuracy
                        tp = conf_matrix[cls, cls]
                        fp = np.sum(conf_matrix[:, cls]) - tp
                        fn = np.sum(conf_matrix[cls, :]) - tp
    
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"  Epoch {epoch+1}/{epochs}: "
                 f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                 f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if rare_classes:
                for cls in rare_classes:
                    if rare_class_total.get(cls, 0) > 0:
                        cls_acc = rare_class_correct[cls] / rare_class_total[cls]
                        print(f"    Class {cls}: Acc: {cls_acc:.4f}, Samples: {rare_class_total[cls]}")
            
            # Check if it is the best model
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
            elif val_loss == best_val_loss and val_acc > best_val_acc:
                is_best = True
            
            if is_best:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                no_improve_epochs = 0
                print(f"    Find the new best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_patience:
                    print(f"    Early stop trigger: {early_stopping_patience} no improvement in epochs")
                    break
        
        # Loading optimal weights
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # Final Assessment
        final_results = self._evaluate_model(model, val_loader, rare_classes)
        final_results.update({
            'training_time': time.time() - start_time,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'model_state': best_model_state
        })
        
        self.results[model_name] = final_results
        return final_results
    
    def _evaluate_model(self, model, data_loader, rare_classes=None):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels, _ in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        report = classification_report(all_labels, all_preds, output_dict=True)

        rare_class_metrics = {}
        if rare_classes:
            for cls in rare_classes:
                if cls < len(conf_matrix):
                    tp = conf_matrix[cls, cls]
                    fp = np.sum(conf_matrix[:, cls]) - tp
                    fn = np.sum(conf_matrix[cls, :]) - tp
                    
                    # 防止除零错误
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    rare_class_metrics[cls] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'rare_class_metrics': rare_class_metrics,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
    
    def run_model_selection(self, train_loader, val_loader, criterion, 
                           optimizer_class=torch.optim.Adam, 
                           optimizer_params_list=None,
                           scheduler_class=None, 
                           scheduler_params=None,
                           epochs=50, early_stopping_patience=10,
                           use_mixed_precision=True,
                           rare_classes=None):
        if optimizer_params_list is None:
            optimizer_params_list = [{'lr': 0.001}] * len(self.models)
        elif isinstance(optimizer_params_list, dict):
            optimizer_params_list = [optimizer_params_list] * len(self.models)
        
        results = {}
        
        for i, (model_name, model) in enumerate(self.models.items()):
            print(f"\n===== train {i+1}/{len(self.models)}: {model_name} =====")
    
            optimizer_params = optimizer_params_list[i % len(optimizer_params_list)]

            model_results = self.train_and_evaluate(
                model_name, 
                train_loader, 
                val_loader, 
                criterion,
                optimizer_class, 
                optimizer_params,
                scheduler_class, 
                scheduler_params,
                epochs, 
                early_stopping_patience,
                use_mixed_precision,
                rare_classes
            )
            
            results[model_name] = model_results

        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        
        print("\n===== Model selection results =====")
        print(fBest model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        for model_name, result in results.items():
            print(f"Model {model_name}: Accuracy = {result['accuracy']:.4f}, Validation loss = {result['best_val_loss']:.4f}")

        return best_model[0], results
    
    def plot_comparison_results(self, output_dir, rare_classes=None):

        import os
        os.makedirs(output_dir, exist_ok=True)

        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        val_losses = [self.results[name]['best_val_loss'] for name in model_names]
        training_times = [self.results[name]['training_time'] / 60 for name in model_names]  # 转换为分钟
        
        # 1. Accuracy comparison chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color='skyblue')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.4f}', ha='center', fontsize=9)
            
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Validation set accuracy')
        plt.ylim(0, max(accuracies) * 1.15)  # Leave space for labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_accuracy_comparison.png'))
        plt.close()
        
        # 2. Comparison of F1 scores for rare categories
        if rare_classes:
            plt.figure(figsize=(12, 8))
            
            # Calculate the F1 score for each rare class
            f1_scores = {}
            for cls in rare_classes:
                f1_scores[cls] = []
                for name in model_names:
                    if cls in self.results[name]['rare_class_metrics']:
                        f1_scores[cls].append(self.results[name]['rare_class_metrics'][cls]['f1'])
                    else:
                        f1_scores[cls].append(0)

            bar_width = 0.8 / len(rare_classes)
            x = np.arange(len(model_names))
 
            for i, cls in enumerate(rare_classes):
                offset = (i - len(rare_classes)/2 + 0.5) * bar_width
                plt.bar(x + offset, f1_scores[cls], width=bar_width, 
                       label=f'Class {cls}')
            
            plt.title('Comparison of F1 scores for rare classes')
            plt.ylabel('F1 score')
            plt.xlabel('Model')
            plt.xticks(x, model_names, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'rare_class_f1_comparison.png'))
            plt.close()
        
        # 3. Training time comparison chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, training_times, color='lightgreen')
    
        for bar, t in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{t:.1f}minutes', ha='center', fontsize=9)
            
        plt.title('Model training time comparison')
        plt.ylabel('Training time (minutes)')
        plt.ylim(0, max(training_times) * 1.15) 
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_training_time_comparison.png'))
        plt.close()
     
        # Confusion Matrix Comparison
        for name in model_names:
            plt.figure(figsize=(8, 6))
            conf_matrix = self.results[name]['confusion_matrix']
            
            try:
                class_names = ['A', 'B', 'C', 'D'][:len(conf_matrix)]
            except:
                class_names = [str(i) for i in range(len(conf_matrix))]
                
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                      xticklabels=class_names,
                      yticklabels=class_names)
            plt.xlabel('Predicted Label')
            plt.ylabel('True label')
            plt.title(f'Model {name} Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'))
            plt.close()
        
        # 6.Speed-Accuracy Tradeoff Chart
        plt.figure(figsize=(10, 8))
    
        sizes = [2000 / (loss + 0.1) for loss in val_losses]
        
        plt.scatter(training_times, accuracies, s=sizes, alpha=0.7)

        for i, name in enumerate(model_names):
            plt.annotate(name, (training_times[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Model performance trade-off: training time vs accuracy')
        plt.xlabel('Training time (minutes)')
        plt.ylabel('Validation accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speed_accuracy_tradeoff.png'))
        plt.close()
        
        print(f"All comparison charts have been saved to {output_dir} ")
