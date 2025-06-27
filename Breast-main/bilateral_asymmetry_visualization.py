"""
双侧不对称性学习可视化
可视化模型如何学习和比较左右乳腺的差异
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
import cv2
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import gaussian_filter


class BilateralAsymmetryVisualizer:
    """可视化双侧不对称性学习过程"""
    
    def __init__(self, model, test_data, output_dir):
        self.model = model
        self.test_data = test_data
        self.output_dir = output_dir
        self.asymmetry_dir = os.path.join(output_dir, 'asymmetry_analysis')
        os.makedirs(self.asymmetry_dir, exist_ok=True)
        
        # 创建中间层模型以提取特征
        self._create_intermediate_models()
    
    def _create_intermediate_models(self):
        """创建用于提取中间特征的模型"""
        # 获取关键层的输出
        layer_outputs = {}
        
        for layer in self.model.model.layers:
            if 'bilateral_asymmetry' in layer.name:
                # 获取不对称层的输入和输出
                self.asymmetry_layer = layer
                
                # 创建一个模型来获取进入不对称层的特征
                self.pre_asymmetry_model = Model(
                    inputs=self.model.model.inputs,
                    outputs=[
                        self.model.model.get_layer('left_attention').output,
                        self.model.model.get_layer('right_attention').output
                    ]
                )
                
                # 获取不对称层内部的中间输出
                # 这需要访问层的内部结构
                break
    
    def extract_asymmetry_features(self, sample_idx):
        """提取不对称性特征"""
        X_sample = [
            self.test_data['bags'][sample_idx:sample_idx+1],
            self.test_data['instance_masks'][sample_idx:sample_idx+1],
            self.test_data['clinical_features'][sample_idx:sample_idx+1],
            self.test_data['side_masks'][sample_idx:sample_idx+1]
        ]
        
        # 获取左右侧的bag特征和注意力
        outputs = self.model.attention_model.predict(X_sample, verbose=0)
        prediction, left_attention, right_attention, left_features, right_features, asymmetry_features = outputs
        
        return {
            'prediction': prediction[0],
            'left_attention': left_attention[0],
            'right_attention': right_attention[0],
            'left_features': left_features[0],
            'right_features': right_features[0],
            'asymmetry_features': asymmetry_features[0],
            'true_label': self.test_data['risk_labels'][sample_idx]
        }
    
    def visualize_feature_differences(self, features_dict, sample_idx):
        """可视化特征差异和不对称性"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        bag_info = self.test_data['bag_info'][sample_idx] if sample_idx < len(self.test_data['bag_info']) else {}
        patient_id = bag_info.get('patient_id', f'Sample_{sample_idx}')
        
        # 标题
        pred_prob = features_dict['prediction']
        pred_label = 1 if pred_prob[1] > 0.5 else 0
        true_label = features_dict['true_label']
        
        fig.suptitle(
            f'Bilateral Asymmetry Analysis - Patient {patient_id}\n'
            f'True: {"High Risk" if true_label else "Medium Risk"} | '
            f'Predicted: {"High Risk" if pred_label else "Medium Risk"} '
            f'(Prob: {pred_prob[1]:.3f})',
            fontsize=16, fontweight='bold'
        )
        
        # 1. 显示左右注意力最高的切片
        bag = self.test_data['bags'][sample_idx]
        instance_mask = self.test_data['instance_masks'][sample_idx]
        side_mask = self.test_data['side_masks'][sample_idx]
        
        left_mask = instance_mask * (1 - side_mask)
        right_mask = instance_mask * side_mask
        
        left_indices = np.where(left_mask > 0)[0]
        right_indices = np.where(right_mask > 0)[0]
        
        # 找到注意力最高的切片
        left_attention_vals = features_dict['left_attention']
        right_attention_vals = features_dict['right_attention']
        
        if len(left_indices) > 0:
            left_max_idx = left_indices[np.argmax([left_attention_vals[idx, 0] for idx in left_indices if idx < len(left_attention_vals)])]
            left_max_attention = left_attention_vals[left_max_idx, 0]
        else:
            left_max_idx = 0
            left_max_attention = 0
        
        if len(right_indices) > 0:
            right_max_idx = right_indices[np.argmax([right_attention_vals[idx, 0] for idx in right_indices if idx < len(right_attention_vals)])]
            right_max_attention = right_attention_vals[right_max_idx, 0]
        else:
            right_max_idx = 0
            right_max_attention = 0
        
        # 显示左侧最高注意力切片
        ax1 = fig.add_subplot(gs[0, 0])
        if left_max_idx < len(bag):
            ax1.imshow(bag[left_max_idx])
            ax1.set_title(f'Left Breast - Highest Attention\nSlice #{left_max_idx}, Attention: {left_max_attention:.3f}')
        ax1.axis('off')
        
        # 显示右侧最高注意力切片
        ax2 = fig.add_subplot(gs[0, 1])
        if right_max_idx < len(bag):
            ax2.imshow(bag[right_max_idx])
            ax2.set_title(f'Right Breast - Highest Attention\nSlice #{right_max_idx}, Attention: {right_max_attention:.3f}')
        ax2.axis('off')
        
        # 显示差异图
        ax3 = fig.add_subplot(gs[0, 2])
        if left_max_idx < len(bag) and right_max_idx < len(bag):
            left_img = bag[left_max_idx]
            right_img = bag[right_max_idx]
            
            # 调整大小以匹配
            if left_img.shape != right_img.shape:
                right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))
            
            # 计算差异
            diff = np.abs(left_img.astype(float) - right_img.astype(float))
            diff = diff / (diff.max() + 1e-8)
            
            im = ax3.imshow(diff, cmap='hot')
            ax3.set_title('Absolute Difference\n(Left - Right)')
            plt.colorbar(im, ax=ax3, fraction=0.046)
        ax3.axis('off')
        
        # 2. 可视化特征向量
        left_feats = features_dict['left_features']
        right_feats = features_dict['right_features']
        
        # 特征直方图比较
        ax4 = fig.add_subplot(gs[1, :])
        x = np.arange(len(left_feats))
        width = 0.35
        
        ax4.bar(x - width/2, left_feats, width, label='Left Features', alpha=0.8, color='blue')
        ax4.bar(x + width/2, right_feats, width, label='Right Features', alpha=0.8, color='red')
        ax4.set_xlabel('Feature Dimension')
        ax4.set_ylabel('Feature Value')
        ax4.set_title('Left vs Right Feature Vectors')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 3. 特征差异分析
        feature_diff = left_feats - right_feats
        abs_diff = np.abs(feature_diff)
        
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(feature_diff, 'g-', linewidth=2)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Feature Dimension')
        ax5.set_ylabel('Left - Right')
        ax5.set_title('Feature Difference (Left - Right)')
        ax5.grid(True, alpha=0.3)
        
        # 4. 显示最大差异的特征
        ax6 = fig.add_subplot(gs[2, 1])
        top_diff_indices = np.argsort(abs_diff)[-10:]  # Top 10 differences
        ax6.barh(range(10), abs_diff[top_diff_indices])
        ax6.set_yticks(range(10))
        ax6.set_yticklabels([f'Feat {i}' for i in top_diff_indices])
        ax6.set_xlabel('Absolute Difference')
        ax6.set_title('Top 10 Most Different Features')
        
        # 5. 不对称性特征可视化
        asymmetry_feats = features_dict['asymmetry_features']
        
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.bar(range(len(asymmetry_feats)), asymmetry_feats, color='purple', alpha=0.7)
        ax7.set_xlabel('Asymmetry Feature Dimension')
        ax7.set_ylabel('Feature Value')
        ax7.set_title('Learned Asymmetry Features')
        ax7.grid(True, alpha=0.3)
        
        # 6. 相关性热图
        ax8 = fig.add_subplot(gs[3, :2])
        
        # 创建特征矩阵
        feature_matrix = np.array([
            left_feats[:20],  # 只显示前20个特征
            right_feats[:20],
            feature_diff[:20],
            asymmetry_feats[:20] if len(asymmetry_feats) >= 20 else np.pad(asymmetry_feats, (0, 20-len(asymmetry_feats)))
        ])
        
        # 计算相关性
        corr_matrix = np.corrcoef(feature_matrix)
        
        sns.heatmap(corr_matrix, 
                   xticklabels=['Left', 'Right', 'Diff', 'Asymm'],
                   yticklabels=['Left', 'Right', 'Diff', 'Asymm'],
                   annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax8)
        ax8.set_title('Feature Correlation Matrix')
        
        # 7. 风险预测贡献度
        ax9 = fig.add_subplot(gs[3, 2])
        
        # 简单的贡献度估计（基于特征值的大小）
        left_contrib = np.mean(np.abs(left_feats))
        right_contrib = np.mean(np.abs(right_feats))
        asymm_contrib = np.mean(np.abs(asymmetry_feats))
        
        contributions = [left_contrib, right_contrib, asymm_contrib]
        contrib_labels = ['Left', 'Right', 'Asymmetry']
        
        ax9.pie(contributions, labels=contrib_labels, autopct='%1.1f%%', startangle=90)
        ax9.set_title('Estimated Feature Contribution')
        
        # 保存图像
        save_path = os.path.join(self.asymmetry_dir, f'asymmetry_analysis_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def create_asymmetry_heatmap(self, sample_idx):
        """创建不对称性热图"""
        features = self.extract_asymmetry_features(sample_idx)
        
        # 获取切片
        bag = self.test_data['bags'][sample_idx]
        instance_mask = self.test_data['instance_masks'][sample_idx]
        side_mask = self.test_data['side_masks'][sample_idx]
        
        left_mask = instance_mask * (1 - side_mask)
        right_mask = instance_mask * side_mask
        
        left_indices = np.where(left_mask > 0)[0]
        right_indices = np.where(right_mask > 0)[0]
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Bilateral Asymmetry Heatmap Analysis', fontsize=16, fontweight='bold')
        
        # 为每对左右切片创建差异热图
        for i in range(min(6, len(left_indices), len(right_indices))):
            row = i // 3
            col = i % 3
            
            left_idx = left_indices[i]
            right_idx = right_indices[i]
            
            if left_idx < len(bag) and right_idx < len(bag):
                left_slice = bag[left_idx]
                right_slice = bag[right_idx]
                
                # 调整大小
                if left_slice.shape != right_slice.shape:
                    right_slice = cv2.resize(right_slice, (left_slice.shape[1], left_slice.shape[0]))
                
                # 转换为灰度
                if len(left_slice.shape) == 3:
                    left_gray = cv2.cvtColor((left_slice * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
                    right_gray = cv2.cvtColor((right_slice * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
                else:
                    left_gray = left_slice
                    right_gray = right_slice
                
                # 计算结构相似性差异
                diff = np.abs(left_gray - right_gray)
                
                # 应用高斯滤波突出主要差异
                diff_smooth = gaussian_filter(diff, sigma=2)
                
                # 创建子图
                ax = fig.add_subplot(gs[row, col])
                im = ax.imshow(diff_smooth, cmap='hot', vmin=0, vmax=0.5)
                ax.set_title(f'Pair {i+1}: L{left_idx} vs R{right_idx}')
                ax.axis('off')
        
        # 添加颜色条
        cbar_ax = fig.add_subplot(gs[:, 3])
        plt.colorbar(im, cax=cbar_ax)
        cbar_ax.set_ylabel('Asymmetry Magnitude', rotation=270, labelpad=20)
        
        save_path = os.path.join(self.asymmetry_dir, f'asymmetry_heatmap_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def generate_all_visualizations(self, num_samples=3):
        """生成所有不对称性分析可视化"""
        print("\n🎨 Generating Bilateral Asymmetry Analysis...")
        
        sample_indices = np.random.choice(
            len(self.test_data['bags']), 
            min(num_samples, len(self.test_data['bags'])), 
            replace=False
        )
        
        generated_files = []
        
        for i, sample_idx in enumerate(sample_indices):
            print(f"\n  Processing sample {i+1}/{len(sample_indices)}...")
            try:
                # 生成特征差异分析
                features = self.extract_asymmetry_features(sample_idx)
                file1 = self.visualize_feature_differences(features, sample_idx)
                generated_files.append(file1)
                print(f"    ✅ Feature analysis saved")
                
                # 生成不对称性热图
                file2 = self.create_asymmetry_heatmap(sample_idx)
                generated_files.append(file2)
                print(f"    ✅ Asymmetry heatmap saved")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Asymmetry analysis completed!")
        print(f"📁 Generated {len(generated_files)} files")
        print(f"📂 Saved to: {self.asymmetry_dir}")
        
        return generated_files


def visualize_bilateral_asymmetry_learning(model, test_data, output_dir):
    """主函数：可视化双侧不对称性学习"""
    print("\n" + "🔬"*30)
    print("Visualizing Bilateral Asymmetry Learning Process")
    print("🔬"*30)
    
    visualizer = BilateralAsymmetryVisualizer(model, test_data, output_dir)
    generated_files = visualizer.generate_all_visualizations(num_samples=3)
    
    print("\n✅ Asymmetry visualization complete!")
    print("   ✨ Feature differences between left and right breast")
    print("   ✨ Learned asymmetry patterns")
    print("   ✨ Contribution analysis for risk prediction")
    
    return generated_files