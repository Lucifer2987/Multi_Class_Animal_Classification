# 🐾 Multi-Class Animal Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Classes](https://img.shields.io/badge/Classes-90%20Animals-purple)
![Model](https://img.shields.io/badge/Model-MobileNetV2-teal)

**A deep learning project that classifies images across 90 different animal species using Transfer Learning with MobileNetV2.**


---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Usage](#-installation--usage)
- [Results](#-results)
- [Animal Classes](#-animal-classes)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🔍 Overview

This project builds a **multi-class image classification** model capable of identifying **90 different animal species** from photographs. It leverages **Transfer Learning** with the **MobileNetV2** architecture pre-trained on ImageNet, fine-tuned for animal classification.

### Key Highlights
- 🧠 **Transfer Learning** with MobileNetV2 (pre-trained on ImageNet)
- 🗂️ **90 animal classes** covering mammals, birds, insects, marine life, and more
- 🔄 **Data Augmentation** to improve generalization
- 📊 **Classification Report** with per-class precision, recall, and F1-score
- ☁️ Runs on **Kaggle** / **Google Colab**

---

## 📦 Dataset

**Source:** [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) by Sourav Banerjee on Kaggle.

```python
import kagglehub
path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
```

| Property | Details |
|----------|---------|
| **Total Classes** | 90 animal species |
| **Format** | Organized by class folders |
| **Source** | Kaggle (iamsouravbanerjee) |
| **Input Size** | 224 × 224 × 3 (RGB) |

---

## 🏗️ Model Architecture

The model uses **MobileNetV2** as a feature extractor with a custom classification head:

```
Input (224×224×3)
    │
    ▼
MobileNetV2 (Pre-trained on ImageNet, weights frozen)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(128, activation='relu')
    │
    ▼
Dropout(0.5)
    │
    ▼
Dense(90, activation='softmax')  ← Output: 90 classes
```

### Why MobileNetV2?
- ⚡ **Lightweight** — efficient for deployment
- 🎯 **High accuracy** — strong ImageNet pre-training
- 📱 **Mobile-friendly** — suitable for edge deployment
- 🔄 **Inverted residuals** — excellent feature extraction

---

## 📁 Project Structure

```
Multi_Class_Animal_Classification/
│
├── Multi Class Animal classification/
│   └── multi_class_animal_C.ipynb    # Main Jupyter Notebook
│
├── README.md                          # Project documentation
└── LICENSE                            # MIT License
```

---

## ✅ Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn
- kagglehub

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Lucifer2987/Multi_Class_Animal_Classification.git
cd Multi_Class_Animal_Classification
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn kagglehub
```

### 3. Run the Notebook

Open the notebook in **Jupyter**, **Google Colab**, or **Kaggle**:

```bash
jupyter notebook "Multi Class Animal classification/multi_class_animal_C.ipynb"
```

### 4. Pipeline Steps

The notebook walks through the following steps:

| Step | Description |
|------|-------------|
| **1. Data Download** | Fetch dataset via `kagglehub` |
| **2. Data Exploration** | List and inspect the 90 animal classes |
| **3. Visualization** | Sample images from each class |
| **4. Preprocessing** | Resize to 224×224, normalize pixel values |
| **5. Augmentation** | `ImageDataGenerator` with rotation, flip, zoom |
| **6. Model Building** | MobileNetV2 base + custom head |
| **7. Training** | Adam optimizer, categorical cross-entropy |
| **8. Evaluation** | Classification report, accuracy metrics |

---

## 📊 Results

The model is evaluated using **scikit-learn's `classification_report`**, providing:

- **Per-class Precision, Recall, F1-Score**
- **Overall Accuracy**
- **Macro and Weighted Averages**

---

## 🦁 Animal Classes

The model classifies **90 animal species**:

| # | Animal | # | Animal | # | Animal |
|---|--------|---|--------|---|--------|
| 1 | Antelope | 31 | Gorilla | 61 | Pig |
| 2 | Badger | 32 | Grasshopper | 62 | Pigeon |
| 3 | Bat | 33 | Hamster | 63 | Porcupine |
| 4 | Bear | 34 | Hare | 64 | Possum |
| 5 | Bee | 35 | Hedgehog | 65 | Raccoon |
| 6 | Beetle | 36 | Hippopotamus | 66 | Rat |
| 7 | Bison | 37 | Hornbill | 67 | Reindeer |
| 8 | Boar | 38 | Horse | 68 | Rhinoceros |
| 9 | Butterfly | 39 | Hummingbird | 69 | Sandpiper |
| 10 | Cat | 40 | Hyena | 70 | Seahorse |
| 11 | Caterpillar | 41 | Jellyfish | 71 | Seal |
| 12 | Chimpanzee | 42 | Kangaroo | 72 | Shark |
| 13 | Cockroach | 43 | Koala | 73 | Sheep |
| 14 | Cow | 44 | Ladybugs | 74 | Snake |
| 15 | Coyote | 45 | Leopard | 75 | Sparrow |
| 16 | Crab | 46 | Lion | 76 | Squid |
| 17 | Crow | 47 | Lizard | 77 | Squirrel |
| 18 | Deer | 48 | Lobster | 78 | Starfish |
| 19 | Dog | 49 | Mouse | 79 | Swan |
| 20 | Dolphin | 50 | Mosquito | 80 | Tiger |
| 21 | Donkey | 51 | Moth | 81 | Turkey |
| 22 | Dragonfly | 52 | Octopus | 82 | Turtle |
| 23 | Duck | 53 | Okapi | 83 | Whale |
| 24 | Eagle | 54 | Orangutan | 84 | Wolf |
| 25 | Elephant | 55 | Otter | 85 | Wombat |
| 26 | Flamingo | 56 | Owl | 86 | Woodpecker |
| 27 | Fly | 57 | Ox | 87 | Zebra |
| 28 | Fox | 58 | Oyster | 88 | Goldfish |
| 29 | Goat | 59 | Panda | 89 | Goose |
| 30 | Gorilla | 60 | Pelecaniformes | 90 | Penguin |


---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |
| **TensorFlow / Keras** | Deep learning framework |
| **MobileNetV2** | Pre-trained feature extractor |
| **ImageDataGenerator** | Data loading & augmentation |
| **scikit-learn** | Evaluation metrics |
| **Matplotlib** | Visualization |
| **NumPy** | Numerical computing |
| **kagglehub** | Dataset download |
| **Kaggle / Colab** | Cloud execution environment |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


Made with ❤️ using TensorFlow & MobileNetV2
