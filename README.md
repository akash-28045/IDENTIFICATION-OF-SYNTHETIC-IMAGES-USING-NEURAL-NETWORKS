# INTRODUCTION:

SmartResNet is a deep learning system designed to detect AI-generated synthetic images using transfer learning with a ResNet50 architecture and resolution upscaling (32×32 → 128×128). Trained on the CIFAKE dataset (120,000 images), it achieves 89.34% validation accuracy through a two-phase training strategy, effectively distinguishing real photographs from AI-generated content to combat misinformation and digital fraud.

# PURPOSE OF THIS PROJECT:

The primary purpose of this project is to create a reliable and scalable detection system for AI-generated synthetic images. In an era where synthetic media is becoming increasingly sophisticated, having robust detection tools is essential for:

Combating Misinformation:  Identifying fake images used in news articles, social media, and propaganda

Preventing Fraud: Detecting synthetic images used in identity theft, financial fraud, or false evidence

Protecting Digital Authenticity: Maintaining trust in digital media across journalism, legal proceedings, and scientific research

Advancing Digital Forensics: Providing tools for investigators to verify the authenticity of visual evidence

Educational Awareness: Understanding the capabilities and limitations of current generative AI models

The system is designed to be accessible to researchers, developers, and forensic analysts, providing a practical solution that can be deployed in various real-world scenarios.

# EXISTING VS ENHANCED FEATURES:

# EXISTING FEATURES:

The existing system proposed by Bird and Lotfi (2024) uses a custom Convolutional Neural Network (CNN) trained on the CIFAKE dataset. The model architecture includes multiple convolutional layers with filter sizes of 16, 32, 64, and 128, followed by dense layers. The best performing model achieved 92.98% accuracy using two layers of 32 filters.

Architecture Pipeline: Input 32×32 RGB image → convolutional layers (with ReLU activation) → max pooling → fully connected layers → sigmoid output → binary classification (Real/Fake).

# KEY CHARACTERISTICS:

Trained from scratch without any pre-trained weights

Shallow architecture with only 4-5 layers

Low resolution input (32×32 pixels)

Limited data augmentation

Single-phase training approach

# CRITICAL LIMITATIONS:

Low Resolution Input: 32×32 images lose fine-grained artifacts and textures essential for distinguishing AI-generated content from real photographs

Shallow Architecture: Limited capacity to learn complex patterns and hierarchical features

No Transfer Learning: Training from scratch without leveraging pre-trained weights from large datasets like ImageNet

Limited Artifact Detection: Cannot effectively capture background inconsistencies and structural anomalies present in AI-generated images

No Fine-tuning Strategy: No mechanism to adapt the model specifically to synthetic image detection

Limited Data Augmentation: Higher risk of overfitting and poor generalization on unseen data

# ENHANCED FEATURES (SMART RESNET):

The proposed SmartResNet system addresses all identified limitations through strategic improvements, combining resolution upscaling, ResNet50 architecture, transfer learning, and fine-tuning into a single unified pipeline.

Architecture Pipeline: Input 128×128×3 RGB image → ResNet50 (50 layers, pre-trained on ImageNet) → Global Average Pooling → Dense (128 units, ReLU) → Dropout (0.2) → Dense (1 unit, Sigmoid) → Binary classification (Real/Fake).

# KEY ENHANCEMENT:

Resolution Upscaling: Images are resized from 32×32 to 128×128 using bilinear interpolation to preserve spatial details, fine-grained artifacts, textures, and background anomalies that are invisible at lower resolutions.

ResNet50 Architecture: A 50-layer residual network with pre-trained ImageNet weights is used for feature extraction. Deep residual connections prevent vanishing gradient problems, enabling capture of hierarchical features from low-level edges to high-level semantic patterns.

Transfer Learning: The model leverages features learned from 1.2 million ImageNet images across 1000 classes, providing rich feature representations out-of-the-box without extensive training from scratch.

Two-Phase Fine-tuning Strategy:

Phase 1 (Frozen Backbone): All ResNet50 layers frozen, custom head trained for 10 epochs with Adam (lr=0.001)

Phase 2 (Fine-tuning): Last 30 layers unfrozen, trained for 10 epochs with reduced learning rate (1e-5) to adapt features specifically to AI-generated image detection

Enhanced Data Augmentation:

Horizontal flipping (random 50%)

Validation split (80/20)

Normalization to [0,1] range

Improved Generalization: Dropout (0.2) regularization and validation monitoring reduce overfitting, improving performance on unseen data.

Performance: Achieves 89.34% validation accuracy on the CIFAKE dataset at 128×128 resolution, effectively detecting diffusion-specific artifacts and background inconsistencies characteristic of AI-generated synthetic images.

# TECH STACK USED:

# CORE FRAMEWORK:

TensorFlow 2.12+

Keras 2.12+

Python 3.8 - 3.11

# LIBRARIES:

NumPy

Matplotlib

Seaborn

Scikit-learn

KaggleHub

# DEVELOPMENT ENVIRONMENT:

Google Colab

NVIDIA T4/V100/A100 GPU

16 GB GPU Memory

# HOW TO RUN THE PROJECT:

Google Colab (Recommended)

# STEP 1: OPEN COLAB AND ENABLE GPU

Go to Google Colab

Click Runtime → Change runtime type → Set Hardware accelerator to GPU


# STEP 2: INSTALL DEPENDENCIES

python
!pip install tensorflow matplotlib seaborn scikit-learn kagglehub


# STEP 3: SET UP KAGGLE API

python
from google.colab import userdata
import os

!mkdir -p ~/.kaggle
KAGGLE_USERNAME = userdata.get('KAGGLE_USERNAME')
KAGGLE_KEY = userdata.get('KAGGLE_KEY')

with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}')
!chmod 600 ~/.kaggle/kaggle.json


# STEP 4: DOWNLOAD DATASET

python
!kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
!unzip cifake-real-and-ai-generated-synthetic-images.zip -d cifake_dataset


# STEP 5: BUILD AND TRAIN MODAL

python
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# LOAD PRE-TRAINED RESNET50:
base_model = applications.ResNet50(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# ADD CUSTOM CLASSIFICATION HEADING:
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# DATA AUGUMENTATION AND LOADING:
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'cifake_dataset/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'cifake_dataset/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# PHASE 1: TRAIN WITH FROZEN BACKBONE (10 epochs)
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# PHASE 2: FINE-TUNE LAST 30 LAYERS(10 epochs)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# SAVE MODAL:
model.save('smartresnet_ai_detection.h5')


# STEP 6: EVALUATE AND VISUALIZE

python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# PLOT ACCURACY:
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# PLOT LOSS:
plt.subplot(1,2,2)
plt.plot(history_fine.history['loss'], label='Train Loss')
plt.plot(history_fine.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# CONFUSION MATRIX:
validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = (Y_pred > 0.5).astype(int).flatten()
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=['FAKE', 'REAL']))

# EXPECTED OUTPUT:


Epoch 1/10 - 45s - loss: 0.4821 - accuracy: 0.8123 - val_loss: 0.3512 - val_accuracy: 0.8734
...

Epoch 10/10 - 42s - loss: 0.2134 - accuracy: 0.8945 - val_loss: 0.1892 - val_accuracy: 0.8934


# FINAL VALIDATION ACCURACY: 89.34%



# INFERENCE: PREDICT ON NEW IMAGE


python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# LOAD TRAINED MODAL:

model = load_model('smartresnet_ai_detection.h5')

# LOAD AND PREPROCESS IMAGE:

img = image.load_img('test_image.jpg', target_size=(128, 128))
img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

# PREDICT

prediction = model.predict(img_array)[0][0]
result = "REAL" if prediction > 0.5 else "FAKE"
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Prediction: {result} (Confidence: {confidence:.2%})")

# KEY FEATURES:

Resolution Upscaling: Processes images at 128×128 resolution to capture fine-grained artifacts

Transfer Learning: Leverages pre-trained ImageNet weights for rich feature extraction

Two-Phase Training: Frozen backbone training followed by fine-tuning for optimal adaptation

Data Augmentation: Horizontal flips and validation splits for improved generalization

Comprehensive Evaluation: Accuracy plots, loss curves, confusion matrix, and classification reports

Visualization: Grad-CAM compatible architecture for explainable AI (future enhancement)

# FUTURE ENHANCEMENTS:
 
Higher Input Resolution: Extending to 224×224 or 512×512 for even finer artifact detection

Vision Transformers: Implementing ViT architecture for global context understanding

Ensemble Methods: Combining multiple models (ResNet50, EfficientNet, ViT) for improved accuracy

Cross-Dataset Testing: Evaluating on images from newer generative models (DALL-E 3, Midjourney, Adobe Firefly)

Explainable AI: Adding Grad-CAM visualization for interpretable predictions

Real-Time Deployment: Web browser extension for social media fake image detection

Video Deepfake Detection: Extending to video processing with temporal analysis

Adversarial Robustness: Training against adversarial attacks

User Interface: Web interface for drag-and-drop image analysis





