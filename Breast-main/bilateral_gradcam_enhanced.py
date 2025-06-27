"""
双侧乳腺GradCAM可视化 - LDCT（低剂量CT）优化版本
针对乳腺LDCT图像的窗宽窗位进行优化
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
import os
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import pydicom
import glob


class BilateralGradCAMLDCTOptimized:
    """针对LDCT优化的双侧GradCAM可视化器"""
    
    def __init__(self, model, test_data, output_dir, dicom_root_dir=None):
        self.model = model
        self.test_data = test_data
        self.output_dir = output_dir
        self.gradcam_dir = os.path.join(output_dir, 'gradcam_ldct_optimized')
        os.makedirs(self.gradcam_dir, exist_ok=True)
        
        # DICOM根目录
        self.dicom_root_dir = dicom_root_dir or 'D:/Desktop/Data_BI-RADS'
        
        # LDCT乳腺图像的典型窗宽窗位设置
        self.breast_window = {
            'center': 50,    # 软组织窗位
            'width': 350     # 软组织窗宽
        }
        
        # 缓存原始图像
        self.original_images_cache = {}
        
    def load_original_dicom_slices(self, patient_id):
        """加载患者的原始DICOM切片"""
        # 尝试不同的路径格式
        possible_paths = [
            os.path.join(self.dicom_root_dir, patient_id),
            os.path.join(self.dicom_root_dir, f"P{patient_id}"),
            os.path.join(self.dicom_root_dir, f"Patient_{patient_id}"),
            os.path.join(self.dicom_root_dir, str(patient_id).zfill(6)),  # 补零格式
        ]
        
        patient_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                patient_dir = path
                break
        
        if patient_dir is None:
            print(f"Warning: DICOM directory not found for patient {patient_id}")
            print(f"  Tried paths: {possible_paths}")
            return None
        
        # 查找所有DICOM文件
        dicom_files = []
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            print(f"Warning: No DICOM files found for patient {patient_id}")
            return None
        
        # 加载和排序DICOM切片
        slices = []
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(file_path)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception as e:
                print(f"Error reading DICOM file {file_path}: {e}")
                continue
        
        # 按照切片位置排序
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0)
        
        # 提取图像数组并应用适合乳腺LDCT的窗口设置
        image_arrays = []
        for ds in slices:
            img = ds.pixel_array.astype(np.float32)
            
            # 应用Rescale Slope和Intercept转换到HU值
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            
            # 应用乳腺软组织窗口
            window_center = self.breast_window['center']
            window_width = self.breast_window['width']
            
            # 如果DICOM文件中有特定的窗口设置，可以选择使用
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                # 检查是否有多个窗口（某些DICOM可能有多个预设窗口）
                if isinstance(ds.WindowCenter, list):
                    # 寻找软组织窗口（通常window center在0-100之间）
                    for i, wc in enumerate(ds.WindowCenter):
                        if 0 <= wc <= 100:
                            window_center = float(wc)
                            window_width = float(ds.WindowWidth[i])
                            break
                else:
                    # 只有当窗位在合理范围内才使用DICOM的设置
                    try:
                        wc_value = float(ds.WindowCenter)
                        ww_value = float(ds.WindowWidth)
                        if -100 <= wc_value <= 200:
                            window_center = wc_value
                            window_width = ww_value
                    except (TypeError, ValueError):
                        # 如果转换失败，使用默认值
                        pass
            
            print(f"  Using window: C={window_center}, W={window_width}")
            
            # 应用窗口
            img_min = window_center - window_width / 2
            img_max = window_center + window_width / 2
            img_windowed = np.clip(img, img_min, img_max)
            
            # 归一化到0-1
            img_normalized = (img_windowed - img_min) / (img_max - img_min)
            
            image_arrays.append(img_normalized)
        
        return image_arrays
    
    def enhance_ldct_for_display(self, ct_image):
        """针对LDCT图像的增强处理"""
        # 确保是numpy数组
        img = np.array(ct_image)
        
        # 如果已经是0-255范围，转换到0-1
        if img.max() > 1:
            img = img / 255.0
        
        # 处理多通道图像
        if len(img.shape) == 3:
            # 如果是RGB且三个通道相同，转换为灰度
            if img.shape[2] == 3 and np.allclose(img[:,:,0], img[:,:,1]) and np.allclose(img[:,:,1], img[:,:,2]):
                img = img[:,:,0]
            else:
                # 转换为灰度
                img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        
        # 转换到uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 确保是单通道
        if len(img_uint8.shape) > 2:
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # 1. 轻微的对比度增强（LDCT不需要太强的增强）
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # 降低clipLimit
        enhanced = clahe.apply(img_uint8)
        
        # 2. 应用轻微的伽马校正改善软组织对比度
        gamma = 0.9  # 轻微提亮
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        # 3. 轻微的去噪（LDCT图像通常噪声较多）
        enhanced = cv2.bilateralFilter(enhanced, 5, 30, 30)
        
        # 转换为RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb
    
    def resize_to_match(self, original_img, target_shape):
        """调整原始图像大小以匹配目标形状"""
        target_h, target_w = target_shape[:2]
        
        # 使用高质量的插值方法
        resized = cv2.resize(original_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def find_matching_slice(self, processed_slice, original_slices):
        """通过图像相似度找到匹配的原始切片"""
        if not original_slices:
            return None
        
        # 将处理后的切片转换为灰度
        if len(processed_slice.shape) == 3:
            processed_gray = cv2.cvtColor((processed_slice * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            processed_gray = (processed_slice * 255).astype(np.uint8)
        
        # 寻找最匹配的切片
        best_match_idx = len(original_slices) // 2  # 默认使用中间切片
        best_score = -1
        
        # 只检查中间60%的切片（通常乳腺组织在这个范围内）
        start_idx = int(len(original_slices) * 0.2)
        end_idx = int(len(original_slices) * 0.8)
        
        for i in range(start_idx, end_idx):
            # 调整原始切片大小以匹配
            original_resized = self.resize_to_match(original_slices[i], processed_slice.shape)
            original_gray = (original_resized * 255).astype(np.uint8)
            if len(original_gray.shape) == 3:
                original_gray = cv2.cvtColor(original_gray, cv2.COLOR_RGB2GRAY)
            
            # 计算结构相似性
            score = cv2.matchTemplate(processed_gray, original_gray, cv2.TM_CCOEFF_NORMED)
            
            if score.max() > best_score:
                best_score = score.max()
                best_match_idx = i
        
        return original_slices[best_match_idx]
    
    def generate_attention_heatmap_on_tissue(self, slice_img, attention_weight):
        """在组织区域生成注意力热图"""
        h, w = slice_img.shape[:2]
        
        # 检测非零区域（组织区域）
        if len(slice_img.shape) == 3:
            gray = cv2.cvtColor((slice_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (slice_img * 255).astype(np.uint8)
        
        # 找到组织区域
        _, tissue_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 创建热图
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if attention_weight > 0.01 and np.sum(tissue_mask) > 0:
            # 找到组织的质心
            contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = w // 2, h // 2
                
                # 获取边界框
                x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
                
                # 创建高斯分布的热图
                Y, X = np.ogrid[:h, :w]
                sigma_x = bbox_w / 4
                sigma_y = bbox_h / 4
                gaussian = np.exp(-((X - cx)**2 / (2 * sigma_x**2) + 
                                   (Y - cy)**2 / (2 * sigma_y**2)))
                
                # 只在组织区域内显示热图
                heatmap = gaussian * (tissue_mask / 255.0) * attention_weight
                
                # 轻微模糊
                heatmap = gaussian_filter(heatmap, sigma=2)
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def create_overlay_with_original_ct(self, processed_slice, heatmap, original_ct, alpha=0.35):
        """在原始CT图像上叠加热图"""
        # 增强原始LDCT图像
        enhanced_ct = self.enhance_ldct_for_display(original_ct)
        
        # 调整大小以匹配处理后的切片
        if enhanced_ct.shape[:2] != processed_slice.shape[:2]:
            enhanced_ct = self.resize_to_match(enhanced_ct, processed_slice.shape)
        
        # 确保是RGB格式
        if len(enhanced_ct.shape) == 2:
            enhanced_ct = cv2.cvtColor(enhanced_ct, cv2.COLOR_GRAY2RGB)
        
        # 创建热图的彩色版本
        if heatmap.max() > 0:
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # 创建alpha通道
            heatmap_alpha = (heatmap > 0.1).astype(np.float32)
            heatmap_alpha = cv2.GaussianBlur(heatmap_alpha, (5, 5), 0)
            adaptive_alpha = alpha * heatmap_alpha[:, :, np.newaxis]
            
            # 叠加
            overlay = enhanced_ct.astype(np.float32)
            overlay = overlay * (1 - adaptive_alpha) + heatmap_colored.astype(np.float32) * adaptive_alpha
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        else:
            overlay = enhanced_ct
        
        return overlay
    
    def extract_attention_and_features(self, sample_idx):
        """提取注意力权重和特征"""
        X_sample = [
            self.test_data['bags'][sample_idx:sample_idx+1],
            self.test_data['instance_masks'][sample_idx:sample_idx+1],
            self.test_data['clinical_features'][sample_idx:sample_idx+1],
            self.test_data['side_masks'][sample_idx:sample_idx+1]
        ]
        
        # 获取注意力输出
        outputs = self.model.attention_model.predict(X_sample, verbose=0)
        prediction, left_attention, right_attention = outputs[:3]
        
        instance_mask = self.test_data['instance_masks'][sample_idx]
        side_mask = self.test_data['side_masks'][sample_idx]
        
        # 分离左右侧
        left_mask = instance_mask * (1 - side_mask)
        right_mask = instance_mask * side_mask
        
        left_indices = np.where(left_mask > 0)[0]
        right_indices = np.where(right_mask > 0)[0]
        
        # 提取注意力权重
        left_weights = {}
        right_weights = {}
        
        if len(left_indices) > 0:
            for i, idx in enumerate(left_indices):
                if idx < len(left_attention[0]):
                    left_weights[idx] = float(left_attention[0, idx, 0])
        
        if len(right_indices) > 0:
            for i, idx in enumerate(right_indices):
                if idx < len(right_attention[0]):
                    right_weights[idx] = float(right_attention[0, idx, 0])
        
        return {
            'prediction': prediction[0],
            'left_indices': left_indices,
            'right_indices': right_indices,
            'left_weights': left_weights,
            'right_weights': right_weights,
            'true_label': self.test_data['risk_labels'][sample_idx]
        }
    
    def visualize_bilateral_gradcam_ldct(self, sample_idx):
        """创建针对LDCT优化的双侧GradCAM可视化"""
        # 获取数据
        attention_data = self.extract_attention_and_features(sample_idx)
        bag = self.test_data['bags'][sample_idx]
        bag_info = self.test_data['bag_info'][sample_idx] if sample_idx < len(self.test_data['bag_info']) else {}
        
        # 获取患者ID
        patient_id = bag_info.get('patient_id', f'Sample_{sample_idx}')
        
        # 加载原始DICOM图像
        print(f"\nLoading original LDCT images for patient {patient_id}...")
        original_slices = self.load_original_dicom_slices(str(patient_id))
        
        if original_slices is None:
            print(f"Warning: Could not load original images for patient {patient_id}")
            use_original = False
        else:
            print(f"Loaded {len(original_slices)} original LDCT slices")
            use_original = True
        
        # 创建图形
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 设置深色背景
        fig.patch.set_facecolor('#f0f0f0')
        
        # 标题
        pred_prob = attention_data['prediction']
        pred_label = 1 if pred_prob[1] > 0.5 else 0
        true_label = attention_data['true_label']
        
        title = f'Bilateral GradCAM on LDCT - Patient {patient_id}\n'
        title += f'True: {"High Risk" if true_label else "Medium Risk"} | '
        title += f'Predicted: {"High Risk" if pred_label else "Medium Risk"} '
        title += f'(Prob: {pred_prob[1]:.3f})'
        
        if use_original:
            title += '\n[Original LDCT with Optimized Window]'
        else:
            title += '\n[Processed Images Only]'
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 选择要显示的切片
        display_slices = []
        
        # 找到左右侧最高注意力切片
        left_max_idx = None
        left_max_weight = 0
        for idx, weight in attention_data['left_weights'].items():
            if weight > left_max_weight:
                left_max_weight = weight
                left_max_idx = idx
        
        right_max_idx = None
        right_max_weight = 0
        for idx, weight in attention_data['right_weights'].items():
            if weight > right_max_weight:
                right_max_weight = weight
                right_max_idx = idx
        
        if left_max_idx is not None:
            display_slices.append(('Left Max', left_max_idx, 'left'))
        if right_max_idx is not None:
            display_slices.append(('Right Max', right_max_idx, 'right'))
        
        # 添加其他高注意力切片
        all_weights = list(attention_data['left_weights'].items()) + list(attention_data['right_weights'].items())
        sorted_weights = sorted(all_weights, key=lambda x: x[1], reverse=True)
        
        for idx, weight in sorted_weights[:12]:
            if idx in attention_data['left_indices']:
                side = 'left'
            else:
                side = 'right'
            
            if (idx, side) not in [(s[1], s[2]) for s in display_slices]:
                display_slices.append((f'{side.title()} #{idx}', idx, side))
            
            if len(display_slices) >= 12:
                break
        
        # 为每个切片创建可视化
        for i, (label, slice_idx, side) in enumerate(display_slices[:12]):
            if i >= 12 or slice_idx >= len(bag):
                continue
            
            row = i // 4
            col = i % 4
            ax = fig.add_subplot(gs[row, col])
            
            # 获取处理后的切片
            processed_slice = bag[slice_idx]
            
            # 获取注意力权重
            if side == 'left':
                weight = attention_data['left_weights'].get(slice_idx, 0)
            else:
                weight = attention_data['right_weights'].get(slice_idx, 0)
            
            # 生成热图
            heatmap = self.generate_attention_heatmap_on_tissue(processed_slice, weight)
            
            # 创建叠加图
            if use_original and original_slices:
                # 使用改进的匹配方法
                original_ct = self.find_matching_slice(processed_slice, original_slices)
                if original_ct is not None:
                    overlay = self.create_overlay_with_original_ct(processed_slice, heatmap, original_ct)
                else:
                    # 如果匹配失败，使用索引匹配
                    original_idx = min(slice_idx, len(original_slices) - 1)
                    original_ct = original_slices[original_idx]
                    overlay = self.create_overlay_with_original_ct(processed_slice, heatmap, original_ct)
            else:
                # 使用处理后的图像
                overlay = self.create_overlay_with_original_ct(processed_slice, heatmap, processed_slice)
            
            # 显示
            ax.imshow(overlay, cmap='gray' if len(overlay.shape) == 2 else None)
            ax.set_title(f'{label} (Slice #{slice_idx})\n'
                        f'Attention: {weight:.3f}',
                        fontsize=10)
            ax.axis('off')
            
            # 根据注意力强度添加边框
            if weight > 0.1:
                edge_color = 'gold'
                edge_width = 3
            elif weight > 0.05:
                edge_color = 'orange'
                edge_width = 2
            else:
                edge_color = 'silver'
                edge_width = 1
                
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(edge_width)
            
            # 添加侧别标记
            ax.text(0.05, 0.95, side.upper()[0], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='red' if side == 'right' else 'blue', 
                           alpha=0.7))
        
        # 保存
        save_path = os.path.join(self.gradcam_dir, f'bilateral_gradcam_ldct_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f0f0f0')
        plt.close()
        
        return save_path
    
    def generate_all_visualizations(self, num_samples=5):
        """生成所有可视化"""
        print("\n🎨 Generating Bilateral GradCAM with LDCT Optimization...")
        print(f"   DICOM directory: {self.dicom_root_dir}")
        print(f"   Using breast tissue window: C={self.breast_window['center']}, W={self.breast_window['width']}")
        
        sample_indices = np.random.choice(
            len(self.test_data['bags']), 
            min(num_samples, len(self.test_data['bags'])), 
            replace=False
        )
        
        generated_files = []
        
        for i, sample_idx in enumerate(sample_indices):
            print(f"\n  Processing sample {i+1}/{len(sample_indices)}...")
            try:
                file_path = self.visualize_bilateral_gradcam_ldct(sample_idx)
                generated_files.append(file_path)
                print(f"    ✅ Saved: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ LDCT-optimized GradCAM visualization completed!")
        print(f"📁 Generated {len(generated_files)} files")
        print(f"📂 Saved to: {self.gradcam_dir}")
        
        return generated_files


def generate_improved_bilateral_gradcam(model, test_data, output_dir, dicom_root_dir='D:/Desktop/Data_BI-RADS'):
    """生成针对LDCT优化的双侧GradCAM可视化"""
    print("\n" + "🔥"*30)
    print("Generating Bilateral GradCAM with LDCT Optimization")
    print("🔥"*30)
    
    visualizer = BilateralGradCAMLDCTOptimized(
        model, 
        test_data, 
        output_dir,
        dicom_root_dir=dicom_root_dir
    )
    
    generated_files = visualizer.generate_all_visualizations(num_samples=5)
    
    print("\n✅ LDCT-optimized visualization complete!")
    print("   ✨ Soft tissue window applied (C=50, W=350)")
    print("   ✨ Enhanced contrast for breast tissue")
    print("   ✨ Reduced noise for better clarity")
    print("   ✨ Attention heatmaps on anatomical structures")
    
    return generated_files