# 🚦 Traffic Sign Recognizer 🚦

## 🏆 Overview
This project is a **Traffic Sign Recognition System** built using **Convolutional Neural Networks (CNN)**. The trained model is integrated into a **Flask** web application, allowing users to upload images of traffic signs and receive real-time predictions. 🖥️

## ✨ Features
- 🧠 **Deep Learning Model:** CNN-based traffic sign classification.
- 🌍 **Dataset:** Trained on 43 traffic sign classes.
- 📊 **Preprocessing:** Image resizing and one-hot encoding of labels.
- 🔬 **Training:** Uses TensorFlow/Keras with dropout layers for regularization.
- 🚀 **Web Interface:** Built with Flask for easy image uploads and predictions.

## 🛠️ Technologies Used
- 🐍 Python
- 🔥 TensorFlow/Keras
- 🔢 NumPy & Pandas
- 📷 OpenCV & PIL
- 📈 Matplotlib
- 🌐 Flask (for web deployment)

## 📂 Dataset
The dataset consists of 43 classes of traffic signs, stored in a folder structure where each subfolder represents a class label. 🏁

📌 **Dataset Link:** [Download Here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## 🏗️ Model Architecture
The model is a **CNN** with the following layers:
- 🎛️ **Conv2D** layers with ReLU activation
- 📉 **MaxPool2D** layers for downsampling
- 🚨 **Dropout** layers to prevent overfitting
- 📏 **Flatten** layer for converting 2D features into 1D
- 🎯 **Dense** layers with Softmax activation for classification

## 🔧 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/JiteshShelke/Traffic-Sign-Recognizer.git
   cd Traffic-Sign-Recognizer
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Train the model (if not using the pre-trained one):
   ```sh
   python train.py
   ```
4. Run the Flask web application:
   ```sh
   python app.py
   ```

## 🎯 Usage
- 📤 Upload an image of a traffic sign via the web app.
- 🤖 The model predicts and displays the class of the traffic sign.

## 📊 Model Training
```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")
```

## 🚀 Future Improvements
- 🔄 Improve accuracy with data augmentation.
- ☁️ Deploy on cloud platforms (AWS/GCP/Heroku).
- 🎥 Add more sign categories and real-time video processing.

## 👨‍💻 Author
[Jitesh Shelke](https://github.com/JiteshShelke) ✨

## 📜 License
This project is licensed under the MIT License 📄.

