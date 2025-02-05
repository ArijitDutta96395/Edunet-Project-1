# CNN Image Classification Project

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify images. The dataset is divided into **train** and **test** folders, and data augmentation techniques are used for better generalization.

## 📂 Dataset
- **Dataset Location:** `Dataset/TRAIN` and `Dataset/TEST`
- Ensure you have images properly categorized into subdirectories based on class labels.
- [Dataset Link] : (#){https://www.kaggle.com/datasets/techsash/waste-classification-data}

## 🚀 Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-username/cnn-image-classification.git
cd cnn-image-classification

# Install dependencies
pip install -r requirements.txt
```

## 🏋️ Training the Model
To train the CNN model, run the following command:
```bash
python train.py
```

## 🛠️ Inference (Making Predictions)
To use the trained model for prediction, run:
```bash
python inference.py --image path_to_image.jpg
```

## 📊 Model Architecture
- **Conv2D + MaxPooling** layers for feature extraction
- **Flatten + Dense** layers for classification
- Dropout layers to prevent overfitting

## 📜 Results
- Model accuracy and loss plots are saved in `/results`
- Predictions can be viewed using `inference.py`

## 🤝 Contributing
Feel free to fork the repository, submit issues, and create pull requests!

## 📜 License
This project is open-source and licensed under the MIT License.

