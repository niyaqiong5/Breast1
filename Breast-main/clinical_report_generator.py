"""
临床报告生成器 - 为医生生成包含AI注意力分析的临床报告
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import logging

logger = logging.getLogger(__name__)

class ClinicalReportGenerator:
    """生成临床AI辅助诊断报告"""
    
    def __init__(self, model, test_data, output_dir):
        self.model = model
        self.test_data = test_data
        self.output_dir = output_dir
        self.report_dir = os.path.join(output_dir, 'clinical_reports')
        os.makedirs(self.report_dir, exist_ok=True)
        
        # 设置样式
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """设置自定义样式"""
        # 标题样式
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # 子标题样式
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2e5090'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # 重要信息样式
        self.styles.add(ParagraphStyle(
            name='Important',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#d32f2f'),
            leftIndent=20
        ))
        
        # 正常文本样式
        self.styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=8
        ))
    
    def generate_patient_report(self, sample_idx, attention_viz_results=None):
        """
        为单个患者生成临床报告
        
        Args:
            sample_idx: 样本索引
            attention_viz_results: 注意力可视化结果（包含图像路径等）
        
        Returns:
            报告文件路径
        """
        try:
            # 获取患者信息
            bag_info = self.test_data['bag_info'][sample_idx]
            patient_id = bag_info['patient_id']
            breast_side = bag_info['breast_side']
            birads_score = bag_info['birads_score']
            
            # 获取预测结果
            X_sample = [
                self.test_data['bags'][sample_idx:sample_idx+1],
                self.test_data['instance_masks'][sample_idx:sample_idx+1],
                self.test_data['clinical_features'][sample_idx:sample_idx+1]
            ]
            
            predictions = self.model.model.predict(X_sample, verbose=0)
            pred_probs = predictions[0]
            pred_class = np.argmax(pred_probs)
            
            # 获取注意力权重
            if hasattr(self.model, 'attention_model'):
                _, attention_weights = self.model.attention_model.predict(X_sample, verbose=0)
                valid_slices = int(np.sum(self.test_data['instance_masks'][sample_idx]))
                attention_scores = attention_weights[0, :valid_slices, 0]
            else:
                attention_scores = None
            
            # 创建PDF文档
            pdf_filename = f'patient_{patient_id}_{breast_side}_clinical_report.pdf'
            pdf_path = os.path.join(self.report_dir, pdf_filename)
            
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # 构建报告内容
            content = []
            
            # 1. 报告标题
            content.append(Paragraph(
                "AI-Assisted Mammography Analysis Report",
                self.styles['CustomTitle']
            ))
            
            # 2. 报告信息
            report_info_data = [
                ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M')],
                ['Patient ID:', str(patient_id)],
                ['Breast Side:', breast_side.title()],
                ['Current BI-RADS:', str(birads_score)],
                ['AI Model:', 'Multi-Instance Learning (MIL) v1.0']
            ]
            
            report_table = Table(report_info_data, colWidths=[2*inch, 3*inch])
            report_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8eaf6')),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f4788')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            content.append(report_table)
            content.append(Spacer(1, 20))
            
            # 3. AI预测结果
            content.append(Paragraph("AI Risk Assessment", self.styles['CustomHeading']))
            
            # 风险等级
            risk_level = "High Risk" if pred_class == 1 else "Low Risk"
            risk_color = colors.HexColor('#d32f2f') if pred_class == 1 else colors.HexColor('#ff9800')
            
            risk_data = [
                ['Predicted Risk Level:', risk_level],
                ['Confidence Score:', f'{pred_probs[pred_class]:.1%}'],
                ['Low Risk Probability:', f'{pred_probs[0]:.1%}'],
                ['High Risk Probability:', f'{pred_probs[1]:.1%}']
            ]
            
            risk_table = Table(risk_data, colWidths=[2.5*inch, 2.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (1, 0), (1, 0), risk_color),
                ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
                ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (1, 0), (1, 0), 12),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8)
            ]))
            content.append(risk_table)
            content.append(Spacer(1, 20))
            
            # 4. 注意力分析
            content.append(Paragraph("Attention Analysis", self.styles['CustomHeading']))
            
            if attention_scores is not None and len(attention_scores) > 0:
                # 注意力统计
                attention_stats = f"""
                The AI model analyzed {len(attention_scores)} image slices. 
                The attention mechanism identified the following key areas:
                """
                content.append(Paragraph(attention_stats, self.styles['CustomNormal']))
                
                # 高注意力切片
                top_slices = np.argsort(attention_scores)[-3:][::-1]
                attention_list = []
                for rank, slice_idx in enumerate(top_slices, 1):
                    attention_list.append([
                        f"Rank {rank}:",
                        f"Slice #{slice_idx + 1}",
                        f"Attention Score: {attention_scores[slice_idx]:.3f}"
                    ])
                
                attention_table = Table(attention_list, colWidths=[1*inch, 1.5*inch, 2.5*inch])
                attention_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
                ]))
                content.append(attention_table)
            else:
                content.append(Paragraph(
                    "Attention analysis not available for this case.",
                    self.styles['CustomNormal']
                ))
            
            content.append(Spacer(1, 20))
            
            # 5. 可视化结果（如果有）
            if attention_viz_results and isinstance(attention_viz_results, dict) and 'overlay_image_path' in attention_viz_results:
                content.append(Paragraph("Visual Analysis", self.styles['CustomHeading']))
                content.append(Paragraph(
                    "The following images show the areas of focus identified by the AI model. "
                    "Red boxes indicate regions that warrant closer examination.",
                    self.styles['CustomNormal']
                ))
                
                # 添加图像
                img_path = attention_viz_results['overlay_image_path']
                if os.path.exists(img_path):
                    img = Image(img_path, width=5*inch, height=3.5*inch)
                    content.append(img)
                    content.append(Spacer(1, 10))
                    content.append(Paragraph(
                        "<i>Figure 1: AI attention overlay on mammography images</i>",
                        self.styles['Caption']
                    ))
            
            content.append(PageBreak())
            
            # 6. 临床建议
            content.append(Paragraph("Clinical Recommendations", self.styles['CustomHeading']))
            
            recommendations = self._generate_recommendations(pred_class, pred_probs[1], birads_score)
            for rec in recommendations:
                content.append(Paragraph(f"• {rec}", self.styles['CustomNormal']))
            
            content.append(Spacer(1, 20))
            
            # 7. 重要说明
            content.append(Paragraph("Important Notice", self.styles['CustomHeading']))
            disclaimer = """
            This report is generated by an AI system designed to assist clinical decision-making. 
            The AI predictions and highlighted areas should be considered as supplementary information 
            and must not replace professional clinical judgment. All findings should be verified 
            through standard clinical and radiological assessment procedures.
            """
            content.append(Paragraph(disclaimer, self.styles['Important']))
            
            # 8. 技术细节（可选）
            content.append(Spacer(1, 20))
            content.append(Paragraph("Technical Details", self.styles['CustomHeading']))
            
            tech_details = f"""
            • Model Type: Multi-Instance Learning with Attention Mechanism
            • Image Processing: {valid_slices if attention_scores is not None else 'N/A'} slices analyzed
            • Feature Extraction: MobileNetV2-based encoder
            • Risk Stratification: Binary classification (Low/High Risk)
            """
            content.append(Paragraph(tech_details, self.styles['CustomNormal']))
            
            # 生成PDF
            doc.build(content)
            
            logger.info(f"✅ 临床报告已生成: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"❌ 生成临床报告失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_recommendations(self, pred_class, high_risk_prob, current_birads):
        """根据预测结果生成临床建议"""
        recommendations = []
        
        if pred_class == 1:  # High Risk
            recommendations.append(
                "The AI model indicates HIGH RISK. Consider additional imaging or biopsy."
            )
            if high_risk_prob > 0.8:
                recommendations.append(
                    "Very high confidence in risk assessment. Immediate follow-up recommended."
                )
        else:  # Low Risk
            recommendations.append(
                "The AI model indicates Low RISK. Standard follow-up protocols apply."
            )
        
        # 基于BI-RADS的建议
        if current_birads >= 4:
            recommendations.append(
                f"Current BI-RADS {current_birads} suggests suspicious findings. "
                "Tissue diagnosis recommended as per ACR guidelines."
            )
        elif current_birads == 3:
            recommendations.append(
                "BI-RADS 3: Probably benign. Short-term follow-up recommended."
            )
        
        # AI与BI-RADS不一致时的建议
        if (pred_class == 1 and current_birads <= 3) or (pred_class == 0 and current_birads >= 4):
            recommendations.append(
                "Note: AI assessment differs from current BI-RADS. "
                "Consider reviewing with multidisciplinary team."
            )
        
        # 通用建议
        recommendations.extend([
            "Review AI-highlighted regions in conjunction with clinical findings.",
            "Consider patient risk factors and clinical history in final assessment.",
            "Document any correlation between AI findings and radiological observations."
        ])
        
        return recommendations
    
    def generate_batch_reports(self, sample_indices, attention_viz_results_dict=None):
        """批量生成多个患者的报告"""
        logger.info(f"📋 开始批量生成 {len(sample_indices)} 份临床报告...")
        
        generated_reports = []
        
        for idx in sample_indices:
            try:
                # 获取对应的可视化结果
                viz_results = None
                if attention_viz_results_dict and idx in attention_viz_results_dict:
                    viz_results = attention_viz_results_dict[idx]
                
                # 生成报告
                report_path = self.generate_patient_report(idx, viz_results)
                if report_path:
                    generated_reports.append(report_path)
                    
            except Exception as e:
                logger.error(f"生成样本 {idx} 的报告失败: {e}")
                continue
        
        # 生成汇总文件
        self._generate_summary_file(generated_reports)
        
        logger.info(f"✅ 成功生成 {len(generated_reports)} 份临床报告")
        return generated_reports
    
    def _generate_summary_file(self, report_paths):
        """生成报告汇总文件"""
        summary_path = os.path.join(self.report_dir, 'reports_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("Clinical Reports Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Total reports: {len(report_paths)}\n\n")
            
            for report_path in report_paths:
                f.write(f"- {os.path.basename(report_path)}\n")
        
        logger.info(f"📄 报告汇总已保存: {summary_path}")