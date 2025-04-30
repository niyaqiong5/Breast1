"""
Breast cancer risk stratification model - model integration
Achieving integrated prediction of multiple models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import logging

logger = logging.getLogger(__name__)

class MultimodalModelWrapper:
    """Wrapping multimodal models to maintain consistent interfaces with traditional ML models"""
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        # Check if X is a tuple (clinical data, image data)
        isinstance(X, tuple) and len(X) == 2:
        X_clinical, X_images = X
        return self.model.predict(X_clinical, X_images)
    
    def predict_proba(self, X):
        isinstance(X, tuple) and len(X) == 2:
        X_clinical, X_images = X
        return self.model.predict_proba(X_clinical, X_images)
  

class EnsembleModel:
    
    def __init__(self, models, weights=None, output_dir=None):
        """
        Initialize model ensemble
        """
        self.models = models
        
        # If no weights are specified, equal weights are used.
        if weights is None:
            self.weights = [1.0 / len(models) for _ in models]
        else:
            # Make sure the number of weights matches the number of models
            if len(weights) != len(models):
                raise ValueError(f"The number of weights ({len(weights)}) must be the same as the number of models ({len(models)})")
            
            # Normalized weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def predict(self, X):
        """
        Use ensemble model for prediction
        """
        # Get the predicted probabilities for each model
        model_probas = []
        
        for i, (name, model) in enumerate(self.models):
            # Pass appropriate inputs depending on the model type
            if name == 'multimodal_dl': 
                proba = model.predict_proba(X)
            else:
                if isinstance(X, tuple):
                    proba = model.predict_proba(X[0])  # Only clinical data are used
                else:
                    proba = model.predict_proba(X)
            model_probas.append(proba)
        
        # Calculate the weighted average probability
        ensemble_proba = np.zeros_like(model_probas[0])
        
        for i, proba in enumerate(model_probas):
            ensemble_proba += proba * self.weights[i]
        
        # Get the most likely category
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred
    
    def predict_proba(self, X):
        """
        Predicted probability
        """
        model_probas = []
        
        for i, (name, model) in enumerate(self.models):
            if name == 'multimodal_dl':
                proba = model.predict_proba(X)
            else:
                # Traditional ML models only require clinical data
                if isinstance(X, tuple):
                    proba = model.predict_proba(X[0])
                else:
                    proba = model.predict_proba(X)
            model_probas.append(proba)

        ensemble_proba = np.zeros_like(model_probas[0])
        
        for i, proba in enumerate(model_probas):
            ensemble_proba += proba * self.weights[i]
        
        return ensemble_proba
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the performance of the ensemble model
        """        
        # Predictions on the test set
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        #Plotting the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['low risk', 'medium risk', 'high risk'],
                   yticklabels=['low risk', 'medium risk', 'high risk'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True label')
        plt.title('Ensemble model confusion matrix')
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'ensemble_confusion_matrix.png'))
        plt.close()
        
        #Calculate the ROC curve for each category
        n_classes = len(np.unique(y_test))
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plotting the ROC Curve
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green']
        for i, color, label in zip(range(n_classes), colors, ['low risk', 'medium risk', 'high risk']):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{label} (AUC = {roc_auc[i]:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble model ROC curve')
        plt.legend(loc="lower right")
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'ensemble_roc_curve.png'))
        plt.close()
        
        # Calculate macro-average AUC
        macro_roc_auc = np.mean(list(roc_auc.values()))
        
        metrics = {
            'accuracy': report['accuracy'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'macro_roc_auc': macro_roc_auc,
            'class_metrics': report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc': {str(k): v for k, v in roc_auc.items()}
        }
        
        # Save evaluation metrics
        if self.output_dir:
            # Save evaluation metrics to text file
            with open(os.path.join(self.output_dir, 'ensemble_evaluation_metrics.txt'), 'w') as f:
                for key, value in metrics.items():
                    if key not in ['class_metrics', 'confusion_matrix', 'roc_auc']:
                        f.write(f"{key}: {value}\n")
        
        return metrics
    
    def evaluate_individual_models(self, X_test, y_test):
        """
        Evaluate the performance of each individual model in the ensemble
        """
        model_metrics = {}
        
        for name, model in self.models:

            y_pred = model.predict(X_test)
            
            accuracy = np.mean(y_pred == y_test)

            report = classification_report(y_test, y_pred, output_dict=True)

            conf_matrix = confusion_matrix(y_test, y_pred)
   
            model_metrics[name] = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': conf_matrix.tolist()
            }
            
            logger.info(f"  Accuracy: {accuracy:.4f}")
        
        # Comparing Model Performance
        model_comparison = pd.DataFrame({
            'Model': [name for name, _ in self.models] + ['Ensemble'],
            'Accuracy': [model_metrics[name]['accuracy'] for name, _ in self.models] + [self.evaluate(X_test, y_test)['accuracy']]
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Accuracy', data=model_comparison)
        plt.title('Model Performance Comparison')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'))
            model_comparison.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        plt.close()
        
        return model_metrics
    
    def optimize_weights(self, X_val, y_val, method='grid_search', n_steps=10):
        """
        Optimize model weights
        """        
        best_accuracy = 0
        best_weights = self.weights.copy()
        
        if method == 'grid_search':
            # 对于两个模型，只需要搜索一个权重
            if len(self.models) == 2:
                logger.info("优化两个模型的权重")
                
                # 生成第一个模型的权重列表
                weight_1_list = np.linspace(0, 1, n_steps)
                
                for weight_1 in weight_1_list:
                    weight_2 = 1 - weight_1
                    weights = [weight_1, weight_2]
                    
                    # 设置新权重
                    self.weights = weights
                    
                    # 评估性能
                    y_pred = self.predict(X_val)
                    accuracy = np.mean(y_pred == y_val)
                    
                    logger.info(f"  权重: {weights}, 准确率: {accuracy:.4f}")
                    
                    # 更新最佳权重
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_weights = weights.copy()
            
            # 对于更多模型，使用网格搜索会变得困难
            else:
                logger.warning("超过两个模型的网格搜索计算成本很高，改用随机搜索")
                method = 'random_search'
        
        if method == 'random_search':
            logger.info("使用随机搜索优化权重")
            
            for _ in range(n_steps):
                # 生成随机权重
                random_weights = np.random.random(len(self.models))
                random_weights = random_weights / np.sum(random_weights)
                
                # 设置新权重
                self.weights = random_weights
                
                # 评估性能
                y_pred = self.predict(X_val)
                accuracy = np.mean(y_pred == y_val)
                
                logger.info(f"  权重: {random_weights}, 准确率: {accuracy:.4f}")
                
                # 更新最佳权重
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = random_weights.copy()
        
        # 设置为最佳权重
        self.weights = best_weights
        
        # 保存最佳权重
        if self.output_dir:
            weight_df = pd.DataFrame({
                'Model': [name for name, _ in self.models],
                'Weight': best_weights
            })
            weight_df.to_csv(os.path.join(self.output_dir, 'optimized_weights.csv'), index=False)
        
        return best_weights
    
    def save(self, filepath):
        """
        Save integrated model configuration
        """
        if not self.output_dir:
            self.output_dir = os.path.dirname(filepath)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        
        # 创建配置字典
        # 将NumPy数组转换为Python列表
        config = {
            'models': [name for name, _ in self.models],
            'weights': [float(w) for w in self.weights]  # 转换为Python原生float类型
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

    
    @classmethod
    def load(cls, config_path, model_dict):
        """
        Load ensemble model from configuration file

        Parameters:
        config_path: configuration file path
        model_dict: model dictionary, key is model name, value is model object

        Return:
        Loaded ensemble model object
        """
        logger.info(f"从 {config_path} 加载集成模型配置")
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 提取模型列表和权重
        model_names = config['models']
        weights = config['weights']
        
        # 确保所有模型都存在
        for name in model_names:
            if name not in model_dict:
                raise ValueError(f"模型 {name} 不在提供的模型字典中")
        
        # 创建模型列表
        models = [(name, model_dict[name]) for name in model_names]
        
        # 创建集成模型
        ensemble = cls(models, weights, os.path.dirname(config_path))
        
        return ensemble
