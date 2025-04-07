# 😊 Emotion Detection Using MobileNet & TensorFlow

A deep learning-based project that uses **Transfer Learning with MobileNet** to classify facial expressions into **seven emotion categories** using the **FER-2013 dataset**. This solution is built using **TensorFlow**, **Keras**, and **Google Colab** for training and evaluation.

---

## 📚 Table of Contents

- [📁 Dataset](#-dataset)
- [🧠 Model Architecture](#-model-architecture)
- [🛠️ Data Preprocessing](#️-data-preprocessing)
- [🏃 Training](#-training)
- [📊 Evaluation & Plots](#-evaluation--plots)
- [🔍 Sample Prediction](#-sample-prediction)
- [📁 Project Structure](#-project-structure)
- [💾 Requirements](#-requirements)
- [🚀 Getting Started](#-getting-started)
- [📬 Contact](#-contact)

---

## 📁 Dataset

We use the **FER-2013 Emotion Detection dataset** consisting of grayscale facial images distributed in 7 emotion classes.

### Emotion Classes:
| Label        | Emotion     |
|--------------|-------------|
| 0            | Angry       |
| 1            | Disgusted   |
| 2            | Fearful     |
| 3            | Happy       |
| 4            | Neutral     |
| 5            | Sad         |
| 6            | Surprised   |

📦 Dataset Directory (after unzip):

```
📂 Emotion detection dataset/
├── 📂 train/
│   ├── angry/
│   ├── disgusted/
│   ├── fearful/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── 📂 test/
    └── (same structure as train)
```

---

## 🧠 Model Architecture

We use **MobileNet** as a base model and add a custom classifier.

```python
base_model = MobileNet(input_shape=(224,224,3), include_top=False)

# Freeze pretrained layers
for layers in base_model.layers:
    layer.trainable = False

# Add classification head
x = Flatten()(base_model.output)
output = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
```

### 🔍 Summary
- Input Shape: `(224, 224, 3)`
- Trainable Parameters: `~351K`
- Total Parameters: `~4.2M`

---

## 🛠️ Data Preprocessing

Using `ImageDataGenerator` for image augmentation and normalization:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('/content/train', target_size=(224,224), batch_size=32)
val_data = val_datagen.flow_from_directory('/content/test', target_size=(224,224), batch_size=32)
```

---

## 🏃 Training

Model is compiled with:

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### ⏳ Callbacks:
- **EarlyStopping** to halt training when validation accuracy stops improving.
- **ModelCheckpoint** to save the best performing model (`best_model.h5`).

```python
es = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
```

### 🚀 Training Execution:

```python
model.fit(train_data,
          validation_data=val_data,
          epochs=30,
          steps_per_epoch=10,
          validation_steps=8,
          callbacks=[es, mc])
```

---

## 📊 Evaluation & Plots

After training, plot the performance:

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title("Accuracy vs Validation Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title("Loss vs Validation Loss")
plt.legend()
plt.show()
```

📈 **(Optional)**: Add screenshots of accuracy and loss plots in your GitHub repo.

---

## 🔍 Sample Prediction

Predict emotion from a sample test image:

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_path = "/content/test/angry/im10.png"
img = load_img(img_path, target_size=(224,224))
i = img_to_array(img) / 255.0
input_arr = np.expand_dims(i, axis=0)

pred = np.argmax(model.predict(input_arr))
emotion = class_indices[pred]

print(f"The predicted emotion is: {emotion}")
```

🖼️ Sample Output:
```
The predicted emotion is neutral
```

---

## 📁 Project Structure

```
📂 Emotion-Detection/
├── 📂 Emotion detection dataset/
├── best_model.h5
├── model_training.ipynb
├── README.md
├── accuracy_plot.png (optional)
└── loss_plot.png (optional)
```

---

## 💾 Requirements

Install these libraries before training:

```bash
pip install tensorflow keras matplotlib numpy pandas
```

---

## 🚀 Getting Started

1. 📥 Clone this repo:
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

2. 📁 Upload and unzip the FER-2013 dataset.

3. ▶️ Run `model_training.ipynb` on (https://colab.research.google.com/github/prashanshi11/Emotion-Detection-opencv/blob/main/Emotion_Detection.ipynb)

4. 🧠 Train the model and save the best version as `best_model.h5`.

5. 🖼️ Try predictions on test images.

---

## 📬 Contact

👨‍💻 Developed by: Prashanshi Yadav  
📧 Email: prashanshiy@gmail.com 
🔗 LinkedIn: (https://www.linkedin.com/in/prashanshi/)  

---

## 🌟 Star this repository!

If you liked this project, don't forget to ⭐ **star** the repo and share it!

---
```

---

✅ Let me know your actual name and LinkedIn/email if you'd like me to personalize the "Contact" section.
