"""
增强的双侧注意力可视化系统
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


class EnhancedBilateralAttentionVisualizer:
    """增强的双侧注意力可视化器"""
    
    def __init__(self, model, test_data, output_dir):
        self.model = model
        self.test_data = test_data
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'bilateral_attention_analysis')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # 设置样式
        self.setup_style()
        
    def setup_style(self):
        """设置可视化样式"""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.dpi': 100
        })
        
        self.colors = {
            'left': '#3498DB',
            'right': '#E74C3C',
            'high_attention': '#FF6B6B',
            'medium_attention': '#4ECDC4',
            'low_attention': '#95A5A6',
            'high_risk': '#E74C3C',
            'medium_risk': '#5DADE2'
        }
    
    def extract_bilateral_attention(self, sample_idx):
        """提取双侧注意力权重"""
        # 准备输入
        X_sample = [
            self.test_data['bags'][sample_idx:sample_idx+1],
            self.test_data['instance_masks'][sample_idx:sample_idx+1],
            self.test_data['clinical_features'][sample_idx:sample_idx+1],
            self.test_data['side_masks'][sample_idx:sample_idx+1]
        ]
        
        # 获取模型输出 - 修复：接收6个输出而不是5个
        outputs = self.model.attention_model.predict(X_sample, verbose=0)
        prediction, left_attention, right_attention, left_features, right_features, asymmetry_features = outputs
        
        # 获取mask信息
        instance_mask = self.test_data['instance_masks'][sample_idx]
        side_mask = self.test_data['side_masks'][sample_idx]
        
        # 计算左右侧mask
        left_mask = instance_mask * (1 - side_mask)
        right_mask = instance_mask * side_mask
        
        # 找到有效索引
        left_indices = np.where(left_mask > 0)[0]
        right_indices = np.where(right_mask > 0)[0]
        
        # 提取有效的注意力权重
        left_att_weights = []
        right_att_weights = []
        
        if len(left_indices) > 0:
            left_att_weights = left_attention[0, left_indices, 0]
            # 重新归一化确保和为1
            if np.sum(left_att_weights) > 0:
                left_att_weights = left_att_weights / np.sum(left_att_weights)
        
        if len(right_indices) > 0:
            right_att_weights = right_attention[0, right_indices, 0]
            # 重新归一化确保和为1
            if np.sum(right_att_weights) > 0:
                right_att_weights = right_att_weights / np.sum(right_att_weights)
        
        return {
            'prediction': prediction[0],
            'left_indices': left_indices,
            'right_indices': right_indices,
            'left_attention': left_att_weights,
            'right_attention': right_att_weights,
            'left_features': left_features[0],
            'right_features': right_features[0],
            'asymmetry_features': asymmetry_features[0]  # 添加这行
        }
    
    def create_comprehensive_bilateral_analysis(self, sample_idx):
        """创建综合的双侧分析可视化"""
        # 提取数据
        attention_data = self.extract_bilateral_attention(sample_idx)
        bag = self.test_data['bags'][sample_idx]
        bag_info = self.test_data['bag_info'][sample_idx] if sample_idx < len(self.test_data['bag_info']) else {}
        true_label = self.test_data['risk_labels'][sample_idx]
        
        # 创建大图
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 标题和基本信息
        patient_id = bag_info.get('patient_id', f'Sample_{sample_idx}')
        pred_prob = attention_data['prediction']
        pred_label = 1 if pred_prob[1] > 0.5 else 0
        
        fig.suptitle(f'Bilateral Breast Analysis - Patient {patient_id}\n'
                     f'True: {"High Risk" if true_label else "Medium Risk"} | '
                     f'Predicted: {"High Risk" if pred_label else "Medium Risk"} '
                     f'(High Risk Prob: {pred_prob[1]:.3f})',
                     fontsize=16, fontweight='bold')
        
        # 2. 左侧乳腺分析（左半部分）
        self._plot_breast_analysis(fig, gs[:2, :3], bag, attention_data, 'left')
        
        # 3. 右侧乳腺分析（右半部分）
        self._plot_breast_analysis(fig, gs[:2, 3:], bag, attention_data, 'right')
        
        # 4. 注意力对比热图
        ax_heatmap = fig.add_subplot(gs[2, :2])
        self._plot_attention_comparison_heatmap(ax_heatmap, attention_data)
        
        # 5. 特征对比
        ax_features = fig.add_subplot(gs[2, 2:4])
        self._plot_feature_comparison(ax_features, attention_data)
        
        # 6. 预测分析
        ax_pred = fig.add_subplot(gs[2, 4:])
        self._plot_prediction_analysis(ax_pred, attention_data, true_label)
        
        # 7. 双侧切片对比展示
        self._plot_bilateral_slices_comparison(fig, gs[3, :], bag, attention_data)
        
        # 保存
        save_path = os.path.join(self.viz_dir, f'bilateral_analysis_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _plot_breast_analysis(self, fig, grid_spec, bag, attention_data, side):
        """绘制单侧乳腺分析"""
        # 获取该侧数据
        if side == 'left':
            indices = attention_data['left_indices']
            attention_weights = attention_data['left_attention']
            color = self.colors['left']
            title = 'Left Breast'
        else:
            indices = attention_data['right_indices']
            attention_weights = attention_data['right_attention']
            color = self.colors['right']
            title = 'Right Breast'
        
        if len(indices) == 0:
            ax = fig.add_subplot(grid_spec)
            ax.text(0.5, 0.5, f'No {side} breast data', 
                ha='center', va='center', fontsize=14)
            ax.set_title(title)
            ax.axis('off')
            return
        
        # Create a sub-gridspec from the provided grid_spec
        # grid_spec is already a SubplotSpec, so we need to create axes differently
        gs_sub = grid_spec.subgridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. 注意力分布条形图
        ax1 = fig.add_subplot(gs_sub[0, :2])
        bars = ax1.bar(range(len(attention_weights)), attention_weights, 
                    color=color, alpha=0.7, edgecolor='black')
        
        # 标记高注意力切片
        max_idx = np.argmax(attention_weights)
        bars[max_idx].set_edgecolor('gold')
        bars[max_idx].set_linewidth(3)
        
        ax1.set_xlabel('Slice Index')
        ax1.set_ylabel('Attention Weight')
        ax1.set_title(f'{title} - Attention Distribution ({len(indices)} slices)')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # 添加平均线
        mean_att = np.mean(attention_weights)
        ax1.axhline(mean_att, color='red', linestyle='--', 
                label=f'Mean: {mean_att:.3f}')
        ax1.legend()
        
        # 2. 最高注意力切片展示
        ax2 = fig.add_subplot(gs_sub[0, 2])
        top_slice_idx = indices[max_idx]
        top_slice = bag[top_slice_idx]
        
        # 创建注意力叠加
        heatmap = self._create_smooth_heatmap(top_slice, attention_weights[max_idx])
        overlay = self._create_attention_overlay(top_slice, heatmap)
        
        ax2.imshow(overlay)
        ax2.set_title(f'Top Attention Slice #{top_slice_idx}\n'
                    f'Weight: {attention_weights[max_idx]:.3f}')
        ax2.axis('off')
        
        # 添加边框
        for spine in ax2.spines.values():
            spine.set_edgecolor('gold')
            spine.set_linewidth(3)
        
        # 3. 注意力权重热力图
        ax3 = fig.add_subplot(gs_sub[1, :])
        
        # 创建切片网格展示
        n_show = min(len(indices), 6)
        sorted_idx = np.argsort(attention_weights)[::-1][:n_show]
        
        for i, idx in enumerate(sorted_idx):
            ax_sub = ax3.inset_axes([i/n_show, 0, 0.9/n_show, 0.9])
            
            slice_idx = indices[idx]
            slice_img = bag[slice_idx]
            att_weight = attention_weights[idx]
            
            # 创建带注意力的叠加图
            heatmap = self._create_smooth_heatmap(slice_img, att_weight)
            overlay = self._create_attention_overlay(slice_img, heatmap)
            
            ax_sub.imshow(overlay)
            ax_sub.set_title(f'#{slice_idx}\n{att_weight:.3f}', fontsize=9)
            ax_sub.axis('off')
            
            # 根据注意力权重设置边框
            edge_color = self._get_attention_color(att_weight, attention_weights)
            for spine in ax_sub.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(2)
        
        ax3.set_title(f'{title} - Top {n_show} Attention Slices')
        ax3.axis('off')
    
    def _plot_attention_comparison_heatmap(self, ax, attention_data):
        """绘制注意力对比热图"""
        # 创建对比矩阵
        max_len = max(len(attention_data['left_attention']), 
                     len(attention_data['right_attention']))
        
        if max_len == 0:
            ax.text(0.5, 0.5, 'No attention data', ha='center', va='center')
            ax.axis('off')
            return
        
        # 创建矩阵
        attention_matrix = np.zeros((2, max_len))
        
        # 填充数据
        if len(attention_data['left_attention']) > 0:
            attention_matrix[0, :len(attention_data['left_attention'])] = attention_data['left_attention']
        
        if len(attention_data['right_attention']) > 0:
            attention_matrix[1, :len(attention_data['right_attention'])] = attention_data['right_attention']
        
        # 绘制热图
        sns.heatmap(attention_matrix, ax=ax, cmap='YlOrRd', 
                   yticklabels=['Left', 'Right'],
                   cbar_kws={'label': 'Attention Weight'})
        
        ax.set_xlabel('Slice Index')
        ax.set_title('Bilateral Attention Comparison Heatmap')
        
        # 添加分割线
        ax.axhline(y=1, color='black', linewidth=2)
    
    def _plot_feature_comparison(self, ax, attention_data):
        """绘制特征对比"""
        left_features = attention_data['left_features']
        right_features = attention_data['right_features']
        
        # 计算特征统计
        left_mean = np.mean(left_features)
        right_mean = np.mean(right_features)
        left_std = np.std(left_features)
        right_std = np.std(right_features)
        
        # 计算差异特征
        diff = np.abs(left_features - right_features)
        diff_mean = np.mean(diff)
        
        # 绘制条形图
        categories = ['Left\nMean', 'Right\nMean', 'Absolute\nDifference']
        values = [left_mean, right_mean, diff_mean]
        errors = [left_std, right_std, np.std(diff)]
        colors_bar = [self.colors['left'], self.colors['right'], 'gray']
        
        bars = ax.bar(categories, values, yerr=errors, capsize=5,
                      color=colors_bar, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, val, err in zip(bars, values, errors):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + err + 0.01,
                   f'{val:.3f}', ha='center', fontweight='bold')
        
        ax.set_ylabel('Feature Value')
        ax.set_title('Bilateral Feature Comparison')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加相似度分数
        similarity = 1 - (diff_mean / (left_mean + right_mean + 1e-8))
        ax.text(0.5, 0.95, f'Similarity Score: {similarity:.3f}',
               transform=ax.transAxes, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
    
    def _plot_prediction_analysis(self, ax, attention_data, true_label):
        """绘制预测分析"""
        pred_probs = attention_data['prediction']
        pred_label = 1 if pred_probs[1] > 0.5 else 0
        
        # 绘制概率条形图
        classes = ['Medium Risk', 'High Risk']
        colors_pred = [self.colors['medium_risk'], self.colors['high_risk']]
        
        bars = ax.bar(classes, pred_probs, color=colors_pred, alpha=0.7,
                      edgecolor='black', linewidth=2)
        
        # 标记正确/错误
        is_correct = (pred_label == true_label)
        title_color = 'green' if is_correct else 'red'
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability')
        ax.set_title('Risk Prediction', color=title_color, fontweight='bold')
        
        # 添加数值标签
        for bar, prob in zip(bars, pred_probs):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + 0.02,
                   f'{prob:.3f}', ha='center', fontweight='bold')
        
        # 添加阈值线
        ax.axhline(0.5, color='red', linestyle='--', label='Threshold')
        
        # 添加正确性标记
        status = '✓ Correct' if is_correct else '✗ Wrong'
        ax.text(0.5, 0.95, status, transform=ax.transAxes,
               ha='center', fontsize=14, fontweight='bold',
               color=title_color,
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='white', 
                        edgecolor=title_color,
                        linewidth=2))
    
    def _plot_bilateral_slices_comparison(self, fig, grid_spec, bag, attention_data):
        """绘制双侧切片对比"""
        ax = fig.add_subplot(grid_spec)
        ax.set_title('Bilateral Slices Comparison - Top 3 Each Side', fontsize=14)
        ax.axis('off')
        
        # 获取左右侧top 3切片
        n_show = 3
        
        # 左侧
        left_slices = []
        if len(attention_data['left_attention']) > 0:
            left_sorted = np.argsort(attention_data['left_attention'])[::-1][:n_show]
            for idx in left_sorted:
                slice_idx = attention_data['left_indices'][idx]
                left_slices.append({
                    'slice': bag[slice_idx],
                    'index': slice_idx,
                    'attention': attention_data['left_attention'][idx]
                })
        
        # 右侧
        right_slices = []
        if len(attention_data['right_attention']) > 0:
            right_sorted = np.argsort(attention_data['right_attention'])[::-1][:n_show]
            for idx in right_sorted:
                slice_idx = attention_data['right_indices'][idx]
                right_slices.append({
                    'slice': bag[slice_idx],
                    'index': slice_idx,
                    'attention': attention_data['right_attention'][idx]
                })
        
        # 创建对比展示
        total_slices = len(left_slices) + len(right_slices)
        if total_slices == 0:
            return
        
        # 左侧切片
        for i, slice_data in enumerate(left_slices):
            x_pos = i / (total_slices + 1)
            ax_sub = ax.inset_axes([x_pos, 0.2, 0.8/total_slices, 0.6])
            
            ax_sub.imshow(slice_data['slice'], cmap='gray' if len(slice_data['slice'].shape) == 2 else None)
            ax_sub.set_title(f"L#{slice_data['index']}\n{slice_data['attention']:.3f}",
                            fontsize=10, color=self.colors['left'])
            ax_sub.axis('off')
            
            # 蓝色边框
            for spine in ax_sub.spines.values():
                spine.set_edgecolor(self.colors['left'])
                spine.set_linewidth(2)
        
        # 分隔线
        if left_slices and right_slices:
            sep_x = len(left_slices) / (total_slices + 1) + 0.4/total_slices
            ax.axvline(x=sep_x, ymin=0.1, ymax=0.9, 
                      color='black', linewidth=2, linestyle='--')
        
        # 右侧切片
        for i, slice_data in enumerate(right_slices):
            x_pos = (len(left_slices) + i + 1) / (total_slices + 1)
            ax_sub = ax.inset_axes([x_pos, 0.2, 0.8/total_slices, 0.6])
            
            ax_sub.imshow(slice_data['slice'], cmap='gray' if len(slice_data['slice'].shape) == 2 else None)
            ax_sub.set_title(f"R#{slice_data['index']}\n{slice_data['attention']:.3f}",
                            fontsize=10, color=self.colors['right'])
            ax_sub.axis('off')
            
            # 红色边框
            for spine in ax_sub.spines.values():
                spine.set_edgecolor(self.colors['right'])
                spine.set_linewidth(2)
        
        # 添加标签
        ax.text(0.25, 0.05, 'LEFT BREAST', transform=ax.transAxes,
               ha='center', fontsize=12, fontweight='bold', color=self.colors['left'])
        ax.text(0.75, 0.05, 'RIGHT BREAST', transform=ax.transAxes,
               ha='center', fontsize=12, fontweight='bold', color=self.colors['right'])
    
    def _create_smooth_heatmap(self, image, attention_weight):
        """创建平滑的注意力热图"""
        h, w = image.shape[:2]
        
        # 创建多个高斯分布
        heatmap = np.zeros((h, w))
        n_gaussians = max(1, int(attention_weight * 5))
        
        for _ in range(n_gaussians):
            # 随机中心，偏向图像中心
            center_y = np.random.normal(h/2, h/6)
            center_x = np.random.normal(w/2, w/6)
            center_y = np.clip(center_y, 0, h-1)
            center_x = np.clip(center_x, 0, w-1)
            
            # 创建高斯
            Y, X = np.ogrid[:h, :w]
            spread = 30 + (1 - attention_weight) * 30
            gaussian = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * spread**2))
            heatmap += gaussian
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap = heatmap * attention_weight
        
        return heatmap
    
    def _create_attention_overlay(self, image, heatmap, alpha=0.4):
        """创建注意力叠加图"""
        # 确保图像是RGB
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image.copy()
        
        # 应用颜色映射到热图
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        
        # 叠加
        overlay = image_rgb * (1 - alpha) + heatmap_colored * alpha
        
        return np.clip(overlay, 0, 1)
    
    def _get_attention_color(self, weight, all_weights):
        """根据注意力权重获取颜色"""
        percentile = np.percentile(all_weights, [33, 67])
        
        if weight > percentile[1]:
            return self.colors['high_attention']
        elif weight > percentile[0]:
            return self.colors['medium_attention']
        else:
            return self.colors['low_attention']
    
    def create_attention_summary_report(self, num_samples=10):
        """创建注意力汇总报告"""
        print("📊 生成双侧注意力汇总报告...")
        
        # 收集所有样本的统计数据
        all_stats = []
        
        sample_indices = np.random.choice(
            len(self.test_data['bags']), 
            min(num_samples, len(self.test_data['bags'])), 
            replace=False
        )
        
        for sample_idx in sample_indices:
            attention_data = self.extract_bilateral_attention(sample_idx)
            
            stats = {
                'sample_idx': sample_idx,
                'left_slices': len(attention_data['left_indices']),
                'right_slices': len(attention_data['right_indices']),
                'left_mean_att': np.mean(attention_data['left_attention']) if len(attention_data['left_attention']) > 0 else 0,
                'right_mean_att': np.mean(attention_data['right_attention']) if len(attention_data['right_attention']) > 0 else 0,
                'left_max_att': np.max(attention_data['left_attention']) if len(attention_data['left_attention']) > 0 else 0,
                'right_max_att': np.max(attention_data['right_attention']) if len(attention_data['right_attention']) > 0 else 0,
                'prediction': attention_data['prediction'][1],  # High risk probability
                'true_label': self.test_data['risk_labels'][sample_idx]
            }
            all_stats.append(stats)
        
        # 创建汇总可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 切片数量分布
        ax = axes[0, 0]
        left_counts = [s['left_slices'] for s in all_stats]
        right_counts = [s['right_slices'] for s in all_stats]
        
        x = np.arange(len(all_stats))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, left_counts, width, label='Left', color=self.colors['left'], alpha=0.7)
        bars2 = ax.bar(x + width/2, right_counts, width, label='Right', color=self.colors['right'], alpha=0.7)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Number of Slices')
        ax.set_title('Slice Count Distribution')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 2. 平均注意力对比
        ax = axes[0, 1]
        left_mean_atts = [s['left_mean_att'] for s in all_stats]
        right_mean_atts = [s['right_mean_att'] for s in all_stats]
        
        ax.scatter(left_mean_atts, right_mean_atts, s=100, alpha=0.6)
        ax.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.5, label='Equal attention')
        
        ax.set_xlabel('Left Mean Attention')
        ax.set_ylabel('Right Mean Attention')
        ax.set_title('Bilateral Mean Attention Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 最大注意力分布
        ax = axes[0, 2]
        data_to_plot = [
            [s['left_max_att'] for s in all_stats],
            [s['right_max_att'] for s in all_stats]
        ]
        
        bp = ax.boxplot(data_to_plot, labels=['Left', 'Right'], patch_artist=True)
        colors_box = [self.colors['left'], self.colors['right']]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Max Attention Weight')
        ax.set_title('Maximum Attention Distribution')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 4. 注意力平衡性
        ax = axes[1, 0]
        balance_scores = []
        for s in all_stats:
            if s['left_mean_att'] > 0 or s['right_mean_att'] > 0:
                balance = abs(s['left_mean_att'] - s['right_mean_att']) / (s['left_mean_att'] + s['right_mean_att'])
                balance_scores.append(balance)
        
        ax.hist(balance_scores, bins=10, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Imbalance Score (0=balanced, 1=imbalanced)')
        ax.set_ylabel('Frequency')
        ax.set_title('Attention Balance Distribution')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 5. 预测准确性 vs 注意力平衡
        ax = axes[1, 1]
        correct_preds = []
        incorrect_preds = []
        
        for s in all_stats:
            pred_label = 1 if s['prediction'] > 0.5 else 0
            is_correct = (pred_label == s['true_label'])
            
            if s['left_mean_att'] > 0 or s['right_mean_att'] > 0:
                balance = abs(s['left_mean_att'] - s['right_mean_att']) / (s['left_mean_att'] + s['right_mean_att'])
                if is_correct:
                    correct_preds.append(balance)
                else:
                    incorrect_preds.append(balance)
        
        if correct_preds:
            ax.hist(correct_preds, bins=8, alpha=0.5, label='Correct', color='green')
        if incorrect_preds:
            ax.hist(incorrect_preds, bins=8, alpha=0.5, label='Incorrect', color='red')
        
        ax.set_xlabel('Attention Imbalance')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Accuracy vs Attention Balance')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 6. 统计摘要
        ax = axes[1, 2]
        ax.axis('off')
        
        # 计算汇总统计
        summary_text = f"""Bilateral Attention Summary
        
Total Samples: {len(all_stats)}

Left Breast:
  Avg slices: {np.mean(left_counts):.1f} ± {np.std(left_counts):.1f}
  Avg attention: {np.mean(left_mean_atts):.3f} ± {np.std(left_mean_atts):.3f}
  Max attention: {np.mean([s['left_max_att'] for s in all_stats]):.3f}

Right Breast:
  Avg slices: {np.mean(right_counts):.1f} ± {np.std(right_counts):.1f}
  Avg attention: {np.mean(right_mean_atts):.3f} ± {np.std(right_mean_atts):.3f}
  Max attention: {np.mean([s['right_max_att'] for s in all_stats]):.3f}

Balance Analysis:
  Avg imbalance: {np.mean(balance_scores):.3f}
  Balanced (<0.2): {sum(b < 0.2 for b in balance_scores)} samples
  Imbalanced (>0.5): {sum(b > 0.5 for b in balance_scores)} samples
"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        fig.suptitle('Bilateral Attention Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.viz_dir, 'bilateral_attention_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 双侧注意力汇总报告已保存: {save_path}")
        
        return save_path
    
    def generate_all_visualizations(self, num_samples=5):
        """生成所有可视化"""
        print("🎨 开始生成双侧注意力可视化...")
        
        # 随机选择样本
        sample_indices = np.random.choice(
            len(self.test_data['bags']), 
            min(num_samples, len(self.test_data['bags'])), 
            replace=False
        )
        
        generated_files = []
        
        # 1. 为每个样本生成综合分析
        for i, sample_idx in enumerate(sample_indices):
            #print(f"\n  📊 分析样本 {i+1}/{num_samples} (index: {sample_idx})...")
            
            try:
                # 生成综合分析
                filepath = self.create_comprehensive_bilateral_analysis(sample_idx)
                generated_files.append(filepath)
                print(f"    ✅ 综合分析完成")
                
            except Exception as e:
                print(f"    ❌ 样本 {sample_idx} 分析失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. 生成汇总报告
        try:
            print("\n  📊 生成汇总报告...")
            summary_path = self.create_attention_summary_report(num_samples=min(20, len(self.test_data['bags'])))
            generated_files.append(summary_path)
            print("    ✅ 汇总报告完成")
        except Exception as e:
            print(f"    ❌ 汇总报告失败: {e}")
        
        print(f"\n🎉 双侧注意力可视化完成！")
        print(f"📁 生成的文件保存在: {self.viz_dir}")
        print(f"   共生成 {len(generated_files)} 个文件")
        
        return self.viz_dir


def visualize_bilateral_model_performance(model, test_data, output_dir):
    """可视化双侧模型性能"""
    print("\n🔍 开始双侧模型性能可视化...")
    
    # 创建可视化器
    visualizer = EnhancedBilateralAttentionVisualizer(model, test_data, output_dir)
    
    # 生成所有可视化
    viz_dir = visualizer.generate_all_visualizations(num_samples=5)
    
    print(f"\n✅ 可视化完成！文件保存在: {viz_dir}")
    
    return viz_dir