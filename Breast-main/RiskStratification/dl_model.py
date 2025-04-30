"""
Breast cancer risk stratification model - Improved deep learning model
Realize multimodal fusion architecture based on pre-trained model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Activation, 
    Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, 
    Concatenate, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import logging
import seaborn as sns

logger = logging.getLogger(__name__)

class ImprovedDLModel:
    def __init__(self, model_params, output_dir):
        self.model_params = model_params
        self.output_dir = output_dir
        self.model = None
        self.clinical_scaler = None
        
        # Get model parameters
        self.backbone = model_params.get('backbone', 'efficientnet')
        self.use_pretrained = model_params.get('use_pretrained', True)
        #self.clinical_hidden_layers = model_params.get('clinical_hidden_layers', [128, 64])
        #self.fusion_hidden_layers = model_params.get('fusion_hidden_layers', [256, 128])
        self.dropout_rate = model_params.get('dropout_rate', 0.3)
        #self.learning_rate = model_params.get('learning_rate', 0.001)
        self.batch_size = model_params.get('batch_size', 16)
        self.epochs = model_params.get('epochs', 50)
        
        # Get fusion strategy from parameters
        #self.fusion_strategy = model_params.get('fusion_strategy', 'concat')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _plot_training_history(self, history):
        """
        Draw the training history curve
        """
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history
        
        plt.figure(figsize=(12, 5))
        
        # Plotting the loss curve
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['loss'], label='Training Loss')
        if 'val_loss' in history_dict:
            plt.plot(history_dict['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Draw the accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history_dict:
            plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close()
    
    def build_model(self, clinical_input_shape, image_input_shape, num_classes=3):
        clinical_input = Input(shape=(clinical_input_shape,), name='clinical_input')
        
        # Very strong regularization
        x_clinical = Dense(64, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.02))(clinical_input)  
        x_clinical = BatchNormalization()(x_clinical)
        x_clinical = Dropout(0.7)(x_clinical) 

        image_input = Input(shape=image_input_shape, name='image_input')
        
        '''  # Using a simpler pre-trained model
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, 
            weights='imagenet', 
            input_tensor=image_input
        )'''

        # Use more powerful pre-trained models
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False, 
            weights='imagenet', 
            input_tensor=image_input
        )
            
        # Freeze all pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Image feature extraction
        x_image = GlobalAveragePooling2D()(base_model.output)
        x_image = Dense(64, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.02))(x_image)
        x_image = BatchNormalization()(x_image)
        x_image = Dropout(0.7)(x_image)

        
        # Feature Fusion 
        combined = Concatenate()([x_clinical, x_image])


        # Fusion layer - adds very strong regularization
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)

        outputs = Dense(num_classes, activation='softmax', name='output')(x)

        model = Model(inputs=[clinical_input, image_input], outputs=outputs)
        
        # Compile the model - use a very low learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
    def preprocess_image_data(self, images, target_size=(224, 224)):
        """
        Optimized image preprocessing functions
        """
        # Make sure the image is a numpy array
        if isinstance(images, list):
            images = np.array(images)
        
        # resize
        processed_images = np.zeros((len(images), target_size[0], target_size[1], 3), dtype=np.float32)
        
        for i, img in enumerate(images):
            # Processing single channel images
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                if len(img.shape) == 3:
                    img = img[:,:,0]
                # Convert to 3 channels
                img = np.stack([img, img, img], axis=-1)
            
            # Ensure valid value range - only process non-zero areas
            nonzero_mask = img > 0
            if np.sum(nonzero_mask) > 0:
                # Normalize the non-zero area
                min_val = np.min(img[nonzero_mask])
                max_val = np.max(img[nonzero_mask])
                
                if max_val > min_val:
                    # Normalized to [0, 1]
                    img_norm = img.copy().astype(np.float32)
                    img_norm[nonzero_mask] = (img[nonzero_mask] - min_val) / (max_val - min_val)
                else:
                    img_norm = img.astype(np.float32) / 255.0
            else:
                img_norm = img.astype(np.float32) / 255.0
                
            # Resize
            from skimage.transform import resize
            resized_img = resize(
                img_norm, 
                (target_size[0], target_size[1]), 
                mode='constant', 
                anti_aliasing=True,
                preserve_range=True
            )
            
            # Make sure the value is in the range [0, 1]
            resized_img = np.clip(resized_img, 0, 1)
            
            # DenseNet Preprocessing
            if self.backbone == 'densenet':
                # Preprocessing formula: (x - mean) / std
                # ImageNet mean and standard deviation
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                # Normalize each channel
                for c in range(3):
                    resized_img[:,:,c] = (resized_img[:,:,c] - mean[c]) / std[c]
                    
            processed_images[i] = resized_img
        
        return processed_images
    
    def train(self, X_clinical, X_images, y, X_val_clinical=None, X_val_images=None, y_val=None):
        """Training multimodal deep learning models"""
        
        # Standardized clinical characteristics
        self.clinical_scaler = StandardScaler()
        X_clinical_scaled = self.clinical_scaler.fit_transform(X_clinical)
        
        # Preprocessing image data
        X_images_processed = self.preprocess_image_data(X_images)
        
        # Processing Authentication Data
        if X_val_clinical is not None and X_val_images is not None and y_val is not None:
            X_val_clinical_scaled = self.clinical_scaler.transform(X_val_clinical)
            X_val_images_processed = self.preprocess_image_data(X_val_images)
            
            validation_data = (
                [X_val_clinical_scaled, X_val_images_processed], 
                to_categorical(y_val)
            )
            validation_split = 0.0
        else:
            validation_data = None
            validation_split = 0.2
        
        # Calculating Class Weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = {i: weight for i, weight in zip(classes, class_weights)}

        y_cat = to_categorical(y)
        
        # Build the model
        if self.model is None:
            # Get the actual number of classes
            actual_num_classes = y_cat.shape[1]
            self.model = self.build_model(
                X_clinical.shape[1], 
                X_images_processed.shape[1:], 
                num_classes=y_cat.shape[1]
            )
        
        # Define callback function 
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ), 
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,  
                patience=7,   
                min_lr=0.0001, 
                verbose=1
            )
            ]
        
        
        # train model
        history = self.model.fit(
            [X_clinical_scaled, X_images_processed], y_cat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save training history
        self._plot_training_history(history)
        
        # Save the model and scaler
        self._save_model()
        
        return history
    
    def predict(self, X_clinical, X_images):
        """
        Use the model to make predictions
        """        
        X_clinical_scaled = self.clinical_scaler.transform(X_clinical)
        
        X_images_processed = self.preprocess_image_data(X_images)
        
        # Predicted class probability
        y_pred_proba = self.model.predict([X_clinical_scaled, X_images_processed])
        
        # Get the most likely class
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred
    
    def predict_proba(self, X_clinical, X_images):
        """
        Predicted probability
        """
        X_clinical_scaled = self.clinical_scaler.transform(X_clinical)
        
        X_images_processed = self.preprocess_image_data(X_images)
        
        return self.model.predict([X_clinical_scaled, X_images_processed])
    
    def evaluate(self, X_clinical, X_images, y, patient_ids=None):    

        X_clinical_scaled = self.clinical_scaler.transform(X_clinical)
        
        X_images_processed = self.preprocess_image_data(X_images)
        
        y_cat = to_categorical(y)
        
        # Evaluate model - output only key metrics
        loss, accuracy = self.model.evaluate(
            [X_clinical_scaled, X_images_processed], y_cat, verbose=0
        )
        
        # predict
        y_pred_proba = self.model.predict([X_clinical_scaled, X_images_processed], verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        report = classification_report(y, y_pred, output_dict=True)
        logger.info("Classification Report:\n" + classification_report(y, y_pred))

        conf_matrix = confusion_matrix(y, y_pred)
        
        # Calculate the ROC curve and AUC for each category
        n_classes = len(np.unique(y))
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            if np.sum(y == i) > 0:  # Make sure the category exists
                fpr[i], tpr[i], _ = roc_curve((y == i).astype(int), y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculate macro-average AUC
        macro_roc_auc = np.mean(list(roc_auc.values()))
        
        # Save the results without printing the details
        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'macro_roc_auc': macro_roc_auc,
            'class_metrics': report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc': {str(k): v for k, v in roc_auc.items()}
        }
        
        return metrics

    def _evaluate_patient_level(self, X_clinical_scaled, X_images_processed, y, patient_ids):
        """
        Patient-level assessment methods within the model
        """
        # Get PID
        unique_pids = list(set(patient_ids))
        
        # Store the prediction results for each patient
        patient_predictions = {}
        patient_true_labels = {}
        
        # p\Predict each slices
        y_pred_proba = self.model.predict([X_clinical_scaled, X_images_processed])
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Group prediction results by patient ID
        for i, pid in enumerate(patient_ids):
            if pid not in patient_predictions:
                patient_predictions[pid] = []
                patient_true_labels[pid] = y[i]
            
            patient_predictions[pid].append(y_pred_proba[i])
        
        # Final prediction at patient level (mean probability)
        patient_final_pred = {}
        patient_final_proba = {}
        for pid, preds in patient_predictions.items():
            #Average the predicted probabilities for all slices
            avg_proba = np.mean(preds, axis=0)
            final_pred = np.argmax(avg_proba)
            patient_final_pred[pid] = final_pred
            patient_final_proba[pid] = avg_proba
        
        # Prepare for Assessment
        y_true_patient = [patient_true_labels[pid] for pid in unique_pids]
        y_pred_patient = [patient_final_pred[pid] for pid in unique_pids]
        y_proba_patient = np.array([patient_final_proba[pid] for pid in unique_pids])
        
        # Calculating evaluation metrics
        accuracy = np.mean(np.array(y_true_patient) == np.array(y_pred_patient))

        report = classification_report(y_true_patient, y_pred_patient, output_dict=True)

        cm = confusion_matrix(y_true_patient, y_pred_patient)
        
        # Calculate ROC curve and AUC (multi-classification case)
        n_classes = len(np.unique(y_true_patient))
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((np.array(y_true_patient) == i).astype(int), y_proba_patient[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plotting the patient-level confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['low', 'medium', 'high'],
                yticklabels=['low', 'medium', 'high'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Patient level confusion matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'patient_level_confusion_matrix.png'))
        plt.close()
        
        # Plotting patient-level ROC curves
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green']
        for i, color, label in zip(range(n_classes), colors, ['low', 'medium', 'high']):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{label} (AUC = {roc_auc[i]:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Patient-level ROC curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'patient_level_roc_curve.png'))
        plt.close()
        
        # Macro-average AUC
        macro_roc_auc = np.mean(list(roc_auc.values()))
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'auc': roc_auc,
            'macro_auc': macro_roc_auc
        }
    

    def _save_model(self):
        """Save the model and scaler to file"""
        if self.model is not None:
            # save model
            model_path = os.path.join(self.output_dir, 'best_model.h5')
            self.model.save(model_path)

            if self.clinical_scaler is not None:
                scaler_path = os.path.join(self.output_dir, 'clinical_scaler.pkl')
                joblib.dump(self.clinical_scaler, scaler_path)
            
            # Save model parameters
            params_path = os.path.join(self.output_dir, 'model_params.json')
            with open(params_path, 'w') as f:
                import json
                json.dump(self.model_params, f, indent=4)

    
    @classmethod
    def load_model(cls, model_path, output_dir=None):
        """
        Load model from file
        """
        if output_dir is None:
            output_dir = os.path.dirname(model_path)

        # Loading model parameters
        params_path = os.path.join(output_dir, 'model_params.json')
        os.path.exists(params_path):
        with open(params_path, 'r') as f:
            import json
            model_params = json.load(f)

        # Creating a Model Instance
        dl_model = cls(model_params, output_dir)
        
        # Loading Keras model
        dl_model.model = load_model(model_path, custom_objects=custom_objects)
        
        scaler_path = os.path.join(output_dir, 'clinical_scaler.pkl')
        os.path.exists(scaler_path):
        dl_model.clinical_scaler = joblib.load(scaler_path)
    
        return dl_model
    
    def predict_single_patient(self, clinical_data, image_data):
        """
        Predict for a single patient
        """
        # Processing clinical data
        if isinstance(clinical_data, dict):
            clinical_array = np.array([clinical_data[feature] for feature in self.model_params.get('feature_names', [])])
        else:
            clinical_array = np.array(clinical_data)
        
        # Make sure the shape is correct
        if clinical_array.ndim == 1:
            clinical_array = clinical_array.reshape(1, -1)
        
        # Standardized clinical characteristics
        if self.clinical_scaler is not None:
            clinical_array = self.clinical_scaler.transform(clinical_array)
        
        # process image data
        # Make sure is 4D: (batch, height, width, channels)
        if image_data.ndim == 3: 
            image_data = np.expand_dims(image_data, axis=0)
        
        # preprocess image
        image_processed = self.preprocess_image_data(image_data)
        
        # predict
        y_pred_proba = self.model.predict([clinical_array, image_processed])
        y_pred = np.argmax(y_pred_proba, axis=1)[0]
        
        return y_pred, y_pred_proba[0]
    
    def explain_prediction(self, clinical_data, image_data, method='grad_cam'):
        """
        Explain model predictions
        """
        if method == 'grad_cam':
            # Implementing Grad-CAM Visualization
            return self._generate_grad_cam(clinical_data, image_data)
        else:
            logger.warning(f"Unsupported interpretation method: {method}")
            return None
    
    def _generate_grad_cam(self, clinical_data, image_data):
        """
        Generate Grad-CAM heatmap
        """
        import tensorflow as tf
        
        # Preprocessing Data
        if isinstance(clinical_data, dict):
            clinical_array = np.array([clinical_data[feature] for feature in self.model_params.get('feature_names', [])])
        else:
            clinical_array = np.array(clinical_data)

        if clinical_array.ndim == 1:
            clinical_array = clinical_array.reshape(1, -1)

        if self.clinical_scaler is not None:
            clinical_array = self.clinical_scaler.transform(clinical_array)

        if image_data.ndim == 3: 
            image_data = np.expand_dims(image_data, axis=0)

        image_processed = self.preprocess_image_data(image_data)
        
        # Get the name of the last convolutional layer
        if self.backbone == 'efficientnet':
            last_conv_layer_name = 'top_activation' 
        elif self.backbone == 'densenet':
            last_conv_layer_name = 'conv5_block16_concat'  
        elif self.backbone == 'resnet':
            last_conv_layer_name = 'conv5_block3_out'
        else:
            logger.warning(f"未Unknown backbone network type: {self.backbone}, unable to create Grad-CAM heatmap")
            return None
        
        # Create a model that returns the output of the last convolutional layer and the original output
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(last_conv_layer_name).output, 
                self.model.output
            ]
        )
        
        # 记录梯度
        with tf.GradientTape() as tape:
            # 计算最后一个卷积层输出和模型预测
            conv_output, predictions = grad_model([clinical_array, image_processed])
            # 获取预测的类别
            pred_class = tf.argmax(predictions[0])
            # 获取该类别的输出
            class_output = predictions[:, pred_class]
        
        # 计算梯度
        grads = tape.gradient(class_output, conv_output)
        
        # 计算导数的全局平均值
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 乘以通道并求和
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = tf.squeeze(heatmap)
        
        # 归一化热图
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # 调整热图大小与原始图像匹配
        import cv2
        heatmap = cv2.resize(heatmap, (image_data.shape[2], image_data.shape[1]))
        
        # 将热图转换为RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 获取原始图像
        original_img = image_data[0]
        
        # 将热图叠加到原始图像上
        if original_img.ndim == 2:  # 如果是灰度图像
            original_img_rgb = np.repeat(original_img[:, :, np.newaxis], 3, axis=2)
        else:
            original_img_rgb = original_img
        
        # 将图像转换为uint8
        if original_img_rgb.dtype != np.uint8:
            original_img_rgb = (original_img_rgb * 255).astype(np.uint8)
        
        # 叠加热图
        superimposed_img = heatmap * 0.4 + original_img_rgb
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return {
            'original_image': original_img,
            'heatmap': heatmap,
            'superimposed_image': superimposed_img,
            'predicted_class': pred_class.numpy(),
            'prediction_probability': float(predictions[0, pred_class])
        }
    
    def analyze_clinical_feature_importance(self, clinical_data, method='permutation'):
    #Analyze the importance of clinical features
        if method == 'permutation':
            from sklearn.inspection import permutation_importance
            
            # Get feature name
            if isinstance(clinical_data, pd.DataFrame):
                feature_names = clinical_data.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(clinical_data.shape[1])]
            
            # Standardized clinical characteristics
            X_clinical = clinical_data.values if isinstance(clinical_data, pd.DataFrame) else clinical_data
            X_clinical_scaled = self.clinical_scaler.transform(X_clinical)
            
            # Create an ad hoc model using only clinical features
            clinical_input = self.model.inputs[0]
            clinical_branch = self.model.layers[1](clinical_input) 
            
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dense) and layer.input.shape[1] == clinical_branch.shape[1]:
                    clinical_output = layer(clinical_branch)
                    break
            
            clinical_model = tf.keras.models.Model(inputs=clinical_input, outputs=clinical_output)
            
            # Calculating feature importance
            result = permutation_importance(
                clinical_model.predict, X_clinical_scaled, 
                n_repeats=10, random_state=42
            )
            
            # Creating a Feature Importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': result.importances_mean,
                'Std': result.importances_std
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            return importance_df.to_dict('records')
    
    def perform_model_analysis(self, dataset, output_path=None):
        """
        Perform model analysis and generate visualizations
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
        
        if output_path is None:
            output_path = self.output_dir
        
        # Get test data
        X_clinical = dataset['test']['clinical']
        X_images = dataset['test']['images']
        y_true = dataset['test']['labels']
        
        # Make predictions
        y_pred = self.predict(X_clinical, X_images)
        y_pred_proba = self.predict_proba(X_clinical, X_images)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate classification report
        cr = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate ROC curve
        n_classes = len(np.unique(y_true))
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Generate visualizations
        # 1. Confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low Risk', 'Medium Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'Medium Risk', 'High Risk'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        confusion_matrix_path = os.path.join(output_path, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        # 2. ROC curve
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green']
        for i, color, label in zip(range(n_classes), colors, ['Low Risk', 'Medium Risk', 'High Risk']):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{label} (AUC = {roc_auc[i]:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        roc_curve_path = os.path.join(output_path, 'roc_curve.png')
        plt.savefig(roc_curve_path)
        plt.close()
        
        # 3. Generate example Grad-CAM visualizations
        # Randomly select samples
        n_samples = min(5, len(X_clinical))
        indices = np.random.choice(len(X_clinical), n_samples, replace=False)
        
        grad_cam_paths = []
        for i, idx in enumerate(indices):
            # Generate Grad-CAM
            result = self._generate_grad_cam(
                X_clinical[idx:idx+1], 
                X_images[idx:idx+1]
            )
            
            if result is not None:
                # Save overlaid images
                grad_cam_path = os.path.join(output_path, f'grad_cam_sample_{i}.png')
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(result['original_image'])
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(result['heatmap'])
                plt.title('Grad-CAM Heatmap')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(result['superimposed_image'])
                plt.title('Overlaid Image')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(grad_cam_path)
                plt.close()
                
                grad_cam_paths.append(grad_cam_path)
        
        # Return analysis results
        result_metrics = {
            'accuracy': cr['accuracy'],
            'confusion_matrix': cm,
            'classification_report': cr,
            'roc_auc': roc_auc,
            'macro_avg_auc': np.mean(list(roc_auc.values())),
            'visualization_paths': {
                'confusion_matrix': confusion_matrix_path,
                'roc_curve': roc_curve_path,
                'grad_cam_samples': grad_cam_paths
            }
        }
        return result_metrics
