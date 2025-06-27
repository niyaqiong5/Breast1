"""
基于模型真实注意力的可视化系统
显示模型实际关注的切片和区域
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns


class TrueAttentionVisualizer:
    """基于真实注意力权重的可视化器"""
    
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.attention_dir = os.path.join(output_dir, 'true_attention_visualizations')
        os.makedirs(self.attention_dir, exist_ok=True)
    
    def create_true_attention_visualization(self, sample_idx, test_data, attention_data):
        """创建基于真实注意力的可视化"""
        
        # 获取数据
        bag = test_data['bags'][sample_idx]
        instance_mask = test_data['instance_masks'][sample_idx]
        side_mask = test_data['side_masks'][sample_idx]
        true_label = test_data['risk_labels'][sample_idx]
        
        # 获取预测
        prediction = attention_data['predictions'][sample_idx]
        pred_prob = prediction[1]
        pred_label = 1 if pred_prob > 0.5 else 0
        
        # 分离左右侧
        left_mask = instance_mask * (1 - side_mask)
        right_mask = instance_mask * side_mask
        
        left_indices = np.where(left_mask > 0)[0]
        right_indices = np.where(right_mask > 0)[0]
        
        # 获取注意力权重
        left_attention = attention_data['left_attention'][sample_idx]
        right_attention = attention_data['right_attention'][sample_idx]
        
        # 创建图形
        fig = plt.figure(figsize=(20, 12))
        
        # 标题
        status = "✓ Correct" if pred_label == true_label else "✗ Wrong"
        status_color = 'green' if pred_label == true_label else 'red'
        
        fig.suptitle(
            f'True Attention Visualization - Patient {sample_idx}\n'
            f'True: {"High Risk" if true_label else "Medium Risk"} | '
            f'Predicted: {"High Risk" if pred_label else "Medium Risk"} '
            f'(Probability: {pred_prob:.3f}) | {status}',
            fontsize=18, color=status_color, fontweight='bold'
        )
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)
        
        # 1. 左侧乳腺可视化
        self._visualize_side_attention(
            fig, gs[:, 0], bag, left_indices, left_attention,
            'Left Breast', 'blue', sample_idx, 'left'
        )
        
        # 2. 右侧乳腺可视化
        self._visualize_side_attention(
            fig, gs[:, 1], bag, right_indices, right_attention,
            'Right Breast', 'red', sample_idx, 'right'
        )
        
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.attention_dir, f'true_attention_patient_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def _visualize_side_attention(self, fig, gridspec, bag, indices, attention_full, 
                                  title, color, sample_idx, side):
        """可视化单侧的真实注意力"""
        
        # 创建子网格
        gs_sub = gridspec.subgridspec(3, 1, height_ratios=[2, 1, 1])
        
        # 1. 显示所有切片及其注意力权重
        ax_slices = fig.add_subplot(gs_sub[0])
        
        if len(indices) > 0:
            # 提取有效的注意力权重
            valid_attentions = []
            valid_indices = []
            
            for idx in indices:
                if idx < len(attention_full):
                    valid_attentions.append(attention_full[idx, 0])
                    valid_indices.append(idx)
            
            if len(valid_attentions) > 0:
                valid_attentions = np.array(valid_attentions)
                valid_indices = np.array(valid_indices)
                
                # 显示前6个切片（按注意力排序）
                n_show = min(6, len(valid_indices))
                sorted_idx = np.argsort(valid_attentions)[::-1][:n_show]
                
                for i, idx in enumerate(sorted_idx):
                    slice_idx = valid_indices[idx]
                    attention = valid_attentions[idx]
                    
                    # 创建子图
                    ax = ax_slices.inset_axes([i/n_show, 0, 0.9/n_show, 0.9])
                    
                    # 显示切片
                    slice_img = bag[slice_idx]
                    if len(slice_img.shape) == 2:
                        ax.imshow(slice_img, cmap='gray')
                    else:
                        ax.imshow(slice_img)
                    
                    # 添加注意力权重作为标题
                    ax.set_title(f'Slice {slice_idx}\nAtt: {attention:.3f}', 
                                fontsize=10, color=color)
                    ax.axis('off')
                    
                    # 根据注意力权重设置边框
                    edge_width = 1 + attention * 10
                    edge_color = 'gold' if attention > np.mean(valid_attentions) else color
                    
                    for spine in ax.spines.values():
                        spine.set_edgecolor(edge_color)
                        spine.set_linewidth(edge_width)
                
                ax_slices.set_title(f'{title} - Top {n_show} Slices by Attention', 
                                   fontsize=14, color=color)
                ax_slices.axis('off')
        else:
            ax_slices.text(0.5, 0.5, f'No {side} breast data', 
                          ha='center', va='center', transform=ax_slices.transAxes)
            ax_slices.axis('off')
        
        # 2. 注意力权重条形图
        ax_bar = fig.add_subplot(gs_sub[1])
        
        if len(indices) > 0 and len(valid_attentions) > 0:
            # 创建条形图
            x_pos = np.arange(len(valid_attentions))
            bars = ax_bar.bar(x_pos, valid_attentions, color=color, alpha=0.7, 
                             edgecolor='black', linewidth=1)
            
            # 标记top切片
            for i in sorted_idx[:n_show]:
                bars[i].set_edgecolor('gold')
                bars[i].set_linewidth(3)
                bars[i].set_alpha(1.0)
            
            # 添加平均线
            mean_att = np.mean(valid_attentions)
            ax_bar.axhline(mean_att, color='red', linestyle='--', 
                          label=f'Mean: {mean_att:.3f}')
            
            ax_bar.set_xlabel('Slice Index')
            ax_bar.set_ylabel('Attention Weight')
            ax_bar.set_title(f'Attention Distribution ({len(valid_attentions)} slices)')
            ax_bar.legend()
            ax_bar.grid(True, alpha=0.3, axis='y')
        
        # 3. 注意力统计信息
        ax_stats = fig.add_subplot(gs_sub[2])
        ax_stats.axis('off')
        
        if len(indices) > 0 and len(valid_attentions) > 0:
            # 计算统计信息
            stats_text = f"""{title} Statistics:
            
Total Slices: {len(valid_indices)}
Mean Attention: {np.mean(valid_attentions):.4f}
Max Attention: {np.max(valid_attentions):.4f}
Min Attention: {np.min(valid_attentions):.4f}
Std Attention: {np.std(valid_attentions):.4f}

Top 3 Slices:"""
            
            for i, idx in enumerate(sorted_idx[:3]):
                stats_text += f"\n  {i+1}. Slice {valid_indices[idx]}: {valid_attentions[idx]:.4f}"
            
            ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', 
                                 edgecolor=color, linewidth=2))
    
    def create_attention_comparison_matrix(self, test_data, attention_data, n_samples=12):
        """创建注意力对比矩阵"""
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        # 随机选择样本
        sample_indices = np.random.choice(len(test_data['bags']), 
                                        size=min(n_samples, len(test_data['bags'])), 
                                        replace=False)
        
        for i, sample_idx in enumerate(sample_indices):
            ax = axes[i]
            
            # 获取数据
            instance_mask = test_data['instance_masks'][sample_idx]
            side_mask = test_data['side_masks'][sample_idx]
            true_label = test_data['risk_labels'][sample_idx]
            pred_prob = attention_data['predictions'][sample_idx, 1]
            
            # 分离左右侧
            left_mask = instance_mask * (1 - side_mask)
            right_mask = instance_mask * side_mask
            
            left_indices = np.where(left_mask > 0)[0]
            right_indices = np.where(right_mask > 0)[0]
            
            # 获取注意力
            left_attention = attention_data['left_attention'][sample_idx]
            right_attention = attention_data['right_attention'][sample_idx]
            
            # 创建注意力矩阵可视化
            max_slices = 20
            attention_matrix = np.zeros((2, max_slices))
            
            # 填充左侧注意力
            for j, idx in enumerate(left_indices[:max_slices]):
                if idx < len(left_attention):
                    attention_matrix[0, j] = left_attention[idx, 0]
            
            # 填充右侧注意力
            for j, idx in enumerate(right_indices[:max_slices]):
                if idx < len(right_attention):
                    attention_matrix[1, j] = right_attention[idx, 0]
            
            # 绘制热力图
            im = ax.imshow(attention_matrix, cmap='hot', aspect='auto')
            
            # 设置标签
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Left', 'Right'])
            ax.set_xlabel('Slice Index')
            
            # 添加标题
            pred_label = 1 if pred_prob > 0.5 else 0
            status = "✓" if pred_label == true_label else "✗"
            title_color = 'green' if pred_label == true_label else 'red'
            
            ax.set_title(f'Patient {sample_idx} {status}\n'
                        f'True: {true_label}, Pred: {pred_prob:.2f}',
                        fontsize=10, color=title_color)
            
            # 添加颜色条
            if i == len(sample_indices) - 1:
                plt.colorbar(im, ax=ax, label='Attention')
        
        # 隐藏多余的子图
        for i in range(len(sample_indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Attention Pattern Comparison Across Patients', fontsize=16)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.attention_dir, 'attention_comparison_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_attention_statistics_plot(self, test_data, attention_data):
        """创建注意力统计分析图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 收集所有注意力数据
        all_left_attentions = []
        all_right_attentions = []
        risk_labels = []
        
        for i in range(len(test_data['bags'])):
            instance_mask = test_data['instance_masks'][i]
            side_mask = test_data['side_masks'][i]
            
            left_mask = instance_mask * (1 - side_mask)
            right_mask = instance_mask * side_mask
            
            left_indices = np.where(left_mask > 0)[0]
            right_indices = np.where(right_mask > 0)[0]
            
            left_attention = attention_data['left_attention'][i]
            right_attention = attention_data['right_attention'][i]
            
            # 提取有效注意力
            for idx in left_indices:
                if idx < len(left_attention):
                    all_left_attentions.append(left_attention[idx, 0])
                    risk_labels.append(test_data['risk_labels'][i])
            
            for idx in right_indices:
                if idx < len(right_attention):
                    all_right_attentions.append(right_attention[idx, 0])
        
        # 1. 注意力分布直方图
        ax = axes[0, 0]
        ax.hist(all_left_attentions, bins=50, alpha=0.6, label='Left', color='blue', density=True)
        ax.hist(all_right_attentions, bins=50, alpha=0.6, label='Right', color='red', density=True)
        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Attention Weights')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 按风险等级的注意力分布
        ax = axes[0, 1]
        
        # 分组数据
        left_att_low = [att for att, risk in zip(all_left_attentions, risk_labels) if risk == 0]
        left_att_high = [att for att, risk in zip(all_left_attentions, risk_labels) if risk == 1]
        
        data_to_plot = []
        labels = []
        
        if left_att_low:
            data_to_plot.append(left_att_low)
            labels.append('Medium Risk')
        if left_att_high:
            data_to_plot.append(left_att_high)
            labels.append('High Risk')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        
        ax.set_ylabel('Attention Weight')
        ax.set_title('Attention Distribution by Risk Level')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 最大注意力 vs 平均注意力
        ax = axes[1, 0]
        
        max_attentions = []
        mean_attentions = []
        
        for i in range(len(test_data['bags'])):
            # 获取该样本的所有注意力
            instance_mask = test_data['instance_masks'][i]
            all_att = []
            
            for j in range(len(instance_mask)):
                if instance_mask[j] > 0:
                    if j < len(attention_data['left_attention'][i]):
                        att = attention_data['left_attention'][i, j, 0]
                        if att > 0:
                            all_att.append(att)
                    elif j < len(attention_data['right_attention'][i]):
                        att = attention_data['right_attention'][i, j, 0]
                        if att > 0:
                            all_att.append(att)
            
            if all_att:
                max_attentions.append(np.max(all_att))
                mean_attentions.append(np.mean(all_att))
        
        if max_attentions and mean_attentions:
            scatter = ax.scatter(mean_attentions, max_attentions, 
                               c=test_data['risk_labels'][:len(mean_attentions)],
                               cmap='RdBu', alpha=0.6, edgecolors='black')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('Mean Attention')
            ax.set_ylabel('Max Attention')
            ax.set_title('Maximum vs Mean Attention per Patient')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Risk Level')
        
        # 4. 注意力集中度分析
        ax = axes[1, 1]
        
        concentration_scores = []
        for i in range(len(test_data['bags'])):
            instance_mask = test_data['instance_masks'][i]
            all_att = []
            
            for j in range(len(instance_mask)):
                if instance_mask[j] > 0:
                    if j < len(attention_data['left_attention'][i]):
                        att = attention_data['left_attention'][i, j, 0]
                        if att > 0:
                            all_att.append(att)
            
            if len(all_att) > 1:
                # 计算基尼系数作为集中度度量
                sorted_att = np.sort(all_att)
                n = len(sorted_att)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_att)) / (n * np.sum(sorted_att)) - (n + 1) / n
                concentration_scores.append(gini)
        
        if concentration_scores:
            ax.hist(concentration_scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax.set_xlabel('Attention Concentration (Gini Coefficient)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Attention Concentration')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加平均值线
            mean_conc = np.mean(concentration_scores)
            ax.axvline(mean_conc, color='red', linestyle='--', 
                      label=f'Mean: {mean_conc:.3f}')
            ax.legend()
        
        plt.suptitle('Attention Weight Statistical Analysis', fontsize=16)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.attention_dir, 'attention_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_all_true_attention_visualizations(self, test_data, attention_data, n_samples=10):
        """生成所有真实注意力可视化"""
        
        print("\n🎯 生成真实注意力可视化...")
        
        generated_files = []
        
        # 1. 个体样本可视化
        sample_indices = np.random.choice(len(test_data['bags']), 
                                        size=min(n_samples, len(test_data['bags'])), 
                                        replace=False)
        
        for i, sample_idx in enumerate(sample_indices):
            print(f"  生成样本 {i+1}/{len(sample_indices)} (index: {sample_idx})...")
            try:
                save_path = self.create_true_attention_visualization(
                    sample_idx, test_data, attention_data
                )
                generated_files.append(save_path)
            except Exception as e:
                print(f"    ❌ 失败: {e}")
        
        # 2. 注意力对比矩阵
        print("  生成注意力对比矩阵...")
        try:
            matrix_path = self.create_attention_comparison_matrix(
                test_data, attention_data
            )
            generated_files.append(matrix_path)
        except Exception as e:
            print(f"    ❌ 对比矩阵失败: {e}")
        
        # 3. 统计分析图
        print("  生成注意力统计分析...")
        try:
            stats_path = self.create_attention_statistics_plot(
                test_data, attention_data
            )
            generated_files.append(stats_path)
        except Exception as e:
            print(f"    ❌ 统计分析失败: {e}")
        
        print(f"\n✅ 真实注意力可视化完成！")
        print(f"📁 生成了 {len(generated_files)} 个文件")
        print(f"📂 保存位置: {self.attention_dir}")
        
        return generated_files


def generate_true_attention_visualizations(model, test_data, output_dir):
    """生成真实注意力可视化的主函数"""
    
    print("\n" + "🎯"*30)
    print("开始生成真实注意力可视化")
    print("🎯"*30)
    
    # 获取模型预测和注意力
    print("\n📊 获取模型预测和注意力权重...")
    
    X_test = [
        test_data['bags'],
        test_data['instance_masks'],
        test_data['clinical_features'],
        test_data['side_masks']
    ]
    
    # 使用注意力模型获取所有输出
    outputs = model.attention_model.predict(X_test, verbose=1)
    
    # 整理注意力数据
    attention_data = {
        'predictions': outputs[0],
        'left_attention': outputs[1],
        'right_attention': outputs[2],
        'left_features': outputs[3],
        'right_features': outputs[4],
        'asymmetry_features': outputs[5]
    }
    
    # 创建真实注意力可视化器
    visualizer = TrueAttentionVisualizer(model, output_dir)
    
    # 生成所有可视化
    generated_files = visualizer.generate_all_true_attention_visualizations(
        test_data, attention_data, n_samples=10
    )
    
    print("\n" + "✅"*30)
    print("真实注意力可视化生成完成！")
    print(f"查看结果: {visualizer.attention_dir}")
    print("✅"*30)
    
    return generated_files