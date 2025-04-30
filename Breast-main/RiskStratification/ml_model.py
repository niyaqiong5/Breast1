"""
Implementing a risk stratification model based on machine learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import logging

logger = logging.getLogger(__name__)

class MLModel:
    
    def __init__(self, model_params, output_dir):

        self.model_params = model_params
        self.output_dir = output_dir
        self.model = None
        self.scaler = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _create_model(self, model_type='random_forest'):
        """
    Create a machine learning model
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 1),
                random_state=self.model_params.get('random_state', 42)
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 3),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 1),
                random_state=self.model_params.get('random_state', 42)
            )
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                C=self.model_params.get('C', 1.0),
                penalty=self.model_params.get('penalty', 'l2'),
                solver=self.model_params.get('solver', 'lbfgs'),
                max_iter=self.model_params.get('max_iter', 1000),
                random_state=self.model_params.get('random_state', 42),
                #multi_class='multinomial'
            )
        elif model_type == 'svm':
            return SVC(
                C=self.model_params.get('C', 1.0),
                kernel=self.model_params.get('kernel', 'rbf'),
                degree=self.model_params.get('degree', 3),
                gamma=self.model_params.get('gamma', 'scale'),
                probability=True,
                random_state=self.model_params.get('random_state', 42)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def train(self, X_train, y_train, model_type='random_forest', optimize_hyperparams=True):
        """
        Train machine learning model
        """
        base_model = self._create_model(model_type)

        from sklearn.impute import SimpleImputer
        
        self.scaler = StandardScaler()
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # 添加这一行来处理缺失值
            ('scaler', self.scaler),
            ('model', base_model)
        ])
        
        #If you need to optimize hyperparameters
        if optimize_hyperparams:
            
            # Defining the parameter grid
            param_grid = self._get_param_grid(model_type)
            
            # 创建网格搜索
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv,
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            # 拟合网格搜索
            grid_search.fit(X_train, y_train)
            
            # 获取最佳模型
            self.model = grid_search.best_estimator_
            
            # 输出最佳参数
            logger.info("最佳参数:")
            for param, value in grid_search.best_params_.items():
                logger.info(f"  {param}: {value}")
        else:
            # 直接训练模型
            self.model = pipeline
            self.model.fit(X_train, y_train)
        
        # 保存模型
        self._save_model()
        
        return self.model
    
    def _get_param_grid(self, model_type):
        """
        Get hyperparameter grid
        """
        if model_type == 'random_forest':
            return {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'gradient_boosting':
            return {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            }
        elif model_type == 'logistic_regression':
            return {
                'model__C': [0.01, 0.1, 1.0, 10.0],
                'model__penalty': ['l2', None],
                'model__solver': ['lbfgs', 'newton-cg']
            }
        elif model_type == 'svm':
            return {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['linear', 'rbf', 'poly'],
                'model__gamma': ['scale', 'auto', 0.1, 0.01]
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def predict(self, X):
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        
        # Predictions on the test set
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculation Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Plotting the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['low risk', 'medium risk', 'high risk'],
                yticklabels=['low risk', 'medium risk', 'high risk'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Get the unique categories present in the data
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        
        if n_classes > 1:
            # 计算每个类别的ROC曲线
            fpr = {}
            tpr = {}
            roc_auc = {}
            
            for i in unique_classes:
                # 确保类别对应的列索引正确
                col_idx = np.where(self.model.classes_ == i)[0]
                if len(col_idx) > 0:
                    col_idx = col_idx[0]
                    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, col_idx])
                    roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 绘制ROC曲线
            plt.figure(figsize=(10, 8))
            class_labels = ['low risk', 'medium risk', 'high risk']
            colors = ['blue', 'red', 'green']
            
            for i, color in zip(unique_classes, colors):
                if i in roc_auc:  # 确保该类别有ROC曲线
                    plt.plot(fpr[i], tpr[i], color=color, lw=2,
                            label=f'{class_labels[i]} (AUC = {roc_auc[i]:0.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-classification ROC curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
            plt.close()
            
            # 计算宏平均AUC
            macro_roc_auc = np.mean(list(roc_auc.values()))
            
            # 将评估指标存储在字典中
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
        
        # 保存评估指标
        with open(os.path.join(self.output_dir, 'evaluation_metrics.txt'), 'w') as f:
            for key, value in metrics.items():
                if key not in ['class_metrics', 'confusion_matrix', 'roc_auc']:
                    f.write(f"{key}: {value}\n")
        
        # 获取特征重要性（如果模型支持）
        feature_names = getattr(self, 'feature_names', [f'feature_{i}' for i in range(X_test.shape[1])])
        self._save_feature_importance(feature_names)
        
        return metrics
    
    def _save_feature_importance(self, feature_names):

        # Get the actual model
        if hasattr(self.model, 'named_steps'):
            model = self.model.named_steps['model']
        else:
            model = self.model
        
        # 检查模型是否有feature_importances_属性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # 确保特征名称列表长度与特征数量匹配
            if len(feature_names) != len(importances):
                logger.warning(f"特征名称数量({len(feature_names)})与特征重要性数量({len(importances)})不匹配，使用默认名称")
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': importances[indices]
            })
            
            # 保存特征重要性
            importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
            
            # 绘制特征重要性
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
            plt.close()
            
            logger.info("已保存特征重要性")
        elif hasattr(model, 'coef_'):
            # 对于线性模型
            coef = model.coef_
            
            # 对于多分类问题，取系数的平均绝对值
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
            
            indices = np.argsort(importances)[::-1]
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Coefficient': importances[indices]
            })
            
            # 保存特征重要性
            importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
            
            # 绘制特征重要性
            plt.figure(figsize=(12, 8))
            plt.title('Feature coefficient')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_coefficients.png'))
            plt.close()
            
            logger.info("已保存特征系数")
    
    def _save_model(self):
        """保存模型到文件"""
        if self.model is not None:
            model_path = os.path.join(self.output_dir, 'model.pkl')
            joblib.dump(self.model, model_path)
            logger.info(f"模型已保存至 {model_path}")
    
    @classmethod
    def load_model(cls, model_path):
        """
        从文件加载模型
        """
        logger.info(f"从 {model_path} 加载模型")
        model = joblib.load(model_path)
        
        # 创建MLModel实例
        ml_model = cls({}, os.path.dirname(model_path))
        ml_model.model = model
        
        return ml_model
