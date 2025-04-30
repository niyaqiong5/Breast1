"""
Breast cancer risk stratification model - Feature selection tool
Responsible for feature selection and dimensionality reduction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging
import os

logger = logging.getLogger(__name__)

class FeatureSelector:
    
    def __init__(self, output_dir):

        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize missing value handler
        self.imputer = SimpleImputer(strategy='mean')
    
    def analyze_feature_correlations(self, X, threshold=0.7):
        """
        Analyze the correlation between features and find highly correlated feature pairs
        """
       
        # Handling missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Calculate the correlation matrix
        corr_matrix = X_imputed.corr()
        
        # Find highly correlated feature pairs
        highly_correlated = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    feature_i = corr_matrix.columns[i]
                    feature_j = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    highly_correlated.append((feature_i, feature_j, correlation))
        
        # Sort by absolute value of correlation coefficient
        highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Plotting correlation heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_correlation.png'))
        plt.close()
        
        if highly_correlated:
            corr_df = pd.DataFrame(highly_correlated, columns=['Feature1', 'Feature2', 'Correlation'])
            corr_df.to_csv(os.path.join(self.output_dir, 'highly_correlated_features.csv'), index=False)
        
        return highly_correlated
    
    def select_features_anova(self, X, y, k=20):
        """
        Feature selection using ANOVA F-value
        """  
        # Handling missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Creating an ANOVA F-value Feature Selector
        selector = SelectKBest(f_classif, k=min(k, X_imputed.shape[1]))
        
        # Fit and transform
        selector.fit(X_imputed, y)
        
        # Get the score for each feature
        scores = selector.scores_
        
        # Creating a Feature Importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores,
            'p-value': selector.pvalues_
        })
        
        # Sort by score
        importance_df = importance_df.sort_values('Score', ascending=False)
        
        # Select the top k features
        selected_features = importance_df.head(k)['Feature'].tolist()
        
        # Save feature importance
        importance_df.to_csv(os.path.join(self.output_dir, 'anova_feature_importance.csv'), index=False)
        
        # Visualizing feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df[:k])), importance_df['Score'][:k], align='center')
        plt.yticks(range(len(importance_df[:k])), importance_df['Feature'][:k])
        plt.xlabel('F-Score')
        plt.title('ANOVA F-Testing feature importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'anova_feature_importance.png'))
        plt.close()

        return selected_features
    
    def select_features_mutual_info(self, X, y, k=20):
        """
        Feature selection using mutual information
        """
        logger.info(f"Select the top {k} features using mutual information...")
        
        # Handling missing values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Creating a Mutual Information Feature Selector
        selector = SelectKBest(mutual_info_classif, k=min(k, X_imputed.shape[1]))
        
        # Fit and transform
        selector.fit(X_imputed, y)
        
        # Get the score for each feature
        scores = selector.scores_
        
        # Creating a Feature Importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores
        })
        
        # Sort by score
        importance_df = importance_df.sort_values('Score', ascending=False)
        
        # Select the top k features
        selected_features = importance_df.head(k)['Feature'].tolist()
        
        # Save feature importance
        importance_df.to_csv(os.path.join(self.output_dir, 'mutual_info_feature_importance.csv'), index=False)
        
        # Visualizing feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df[:k])), importance_df['Score'][:k], align='center')
        plt.yticks(range(len(importance_df[:k])), importance_df['Feature'][:k])
        plt.xlabel('Mutual Information Score')
        plt.title('Mutual Information Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mutual_info_feature_importance.png'))
        plt.close()

        return selected_features
    
    def select_features_random_forest(self, X, y, k=20):
        """
        Feature selection using random forest
        """    
    
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Creating a Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

         # Directly using feature importance of random forest
        rf.fit(X_imputed, y)
        feature_importances = rf.feature_importances_
            
        # Creating a Feature Importance DataFrame
        importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importances
            })
            
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
            
        # Select the top k features
        selected_features = importance_df.head(k)['Feature'].tolist()
            
        # Save feature importance
        importance_df.to_csv(os.path.join(self.output_dir, 'rf_feature_importance.csv'), index=False)
            
        # Visualizing feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df[:k])), importance_df['Importance'][:k], align='center')
        plt.yticks(range(len(importance_df[:k])), importance_df['Feature'][:k])
        plt.xlabel('Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rf_feature_importance.png'))
        plt.close()
  
        return selected_features
    
    
    def apply_pca(self, X, n_components=None, variance_threshold=0.95):
        """
        Apply PCA dimensionality reduction
        """   
  
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        if n_components is None:
            pca = PCA(n_components=variance_threshold, svd_solver='full')
        else:
            pca = PCA(n_components=min(n_components, X_imputed.shape[1]), svd_solver='full')
 
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate the proportion of variance explained by each principal component
        explained_variance_ratio = pca.explained_variance_ratio_
        
        # Calculate the cumulative explained variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Determine the number of principal components needed
        if n_components is None:
            n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        
        # Visualizing Cumulative Explained Variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
        plt.axhline(y=variance_threshold, color='r', linestyle='--')
        plt.axvline(x=n_components, color='g', linestyle='--')
        plt.xlabel('Number of principal components')
        plt.ylabel('Cumulative explained variance proportion')
        plt.title('PCA Cumulative Explained Variance')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'pca_variance.png'))
        plt.close()
        
        # Visualizing the first two principal components
        if X_pca.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca[:, 0], X_pca[:, 1])
            plt.xlabel(f'Principal Components 1 ({explained_variance_ratio[0]:.2%})')
            plt.ylabel(f'Principal Components 2 ({explained_variance_ratio[1]:.2%})')
            plt.title('PCA: The first two principal components')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'pca_scatter.png'))
            plt.close()
        
        #Save the explained variance of the principal components
        variance_df = pd.DataFrame({
            'Principal Component': range(1, len(explained_variance_ratio) + 1),
            'Explained Variance Ratio': explained_variance_ratio,
            'Cumulative Explained Variance Ratio': cumulative_variance_ratio
        })
        variance_df.to_csv(os.path.join(self.output_dir, 'pca_variance.csv'), index=False)
        
        # Preserve feature contributions to principal components
        if hasattr(pca, 'components_'):
            components_df = pd.DataFrame(
                pca.components_[:n_components, :].T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=X.columns
            )
            components_df.to_csv(os.path.join(self.output_dir, 'pca_components.csv'))
        
        logger.info(f"After PCA dimensionality reduction, {n_components} principal components are retained, explaining the variance {cumulative_variance_ratio[n_components-1]:.2%}")

        return X_pca[:, :n_components], pca, scaler
    
    def ensemble_feature_selection(self, X, y, k=20, methods=None):
        """
        Integrate multiple feature selection methods
        """
        if methods is None:
            methods = ['anova', 'mutual_info', 'random_forest']
        
        # Check if there are NaN values ​​in the data
        if X.isnull().values.any():        
        selected_features = {}
        
        # Select features using various methods
        if 'anova' in methods:
            selected_features['anova'] = set(self.select_features_anova(X, y, k))
        
        if 'mutual_info' in methods:
            selected_features['mutual_info'] = set(self.select_features_mutual_info(X, y, k))
        
        if 'random_forest' in methods:

            selected_features['random_forest'] = set(self.select_features_random_forest(X, y, k))
        
        # Count the number of times each feature is selected
        feature_counts = {}
        for method, features in selected_features.items():
            for feature in features:
                if feature not in feature_counts:
                    feature_counts[feature] = 0
                feature_counts[feature] += 1
        
        # If no features are selected, return the first k column names
        if not feature_counts:
            return list(X.columns[:min(k, len(X.columns))])
        
        # Create a feature count DataFrame
        count_df = pd.DataFrame({
            'Feature': list(feature_counts.keys()),
            'Count': list(feature_counts.values())
        })
        
        # 按计数排序
        count_df = count_df.sort_values('Count', ascending=False)
        
        # 保存特征计数
        count_df.to_csv(os.path.join(self.output_dir, 'ensemble_feature_counts.csv'), index=False)
        
        # 选择被至少一半方法选中的特征
        threshold = max(1, len(methods) / 2)  # 确保至少需要1个方法选中
        ensemble_features = count_df[count_df['Count'] >= threshold]['Feature'].tolist()
        
        # 如果选择的特征太少，则选择计数最高的k个特征
        if len(ensemble_features) < k:
            ensemble_features = count_df.head(min(k, len(count_df)))['Feature'].tolist()
        
        # 可视化特征选择结果
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(count_df[:min(k, len(count_df))])), count_df['Count'][:min(k, len(count_df))], align='center')
        plt.yticks(range(len(count_df[:min(k, len(count_df))])), count_df['Feature'][:min(k, len(count_df))])
        plt.xlabel('Number of times selected')
        plt.title('Integrate feature selection results')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ensemble_feature_selection.png'))
        plt.close()
        
        logger.info(f"集成特征选择选择了{len(ensemble_features)}个特征")
        return ensemble_features
