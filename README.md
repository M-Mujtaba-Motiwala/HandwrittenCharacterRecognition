# ✍️ Handwritten Character Recognition Using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains an end-to-end **Handwritten Character and Digit Recognition** project developed as part of the **CodeAlpha Artificial Intelligence Internship**.

The system uses a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras** to classify grayscale images of handwritten digits from **0 to 9**.

## 📁 Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
├── CodeAlpha_HandwrittenCharacterRecognition/
│   ├── CodeAlpha_HandwrittenCharacterRecognition.ipynb  # Interactive Jupyter Notebook
│   ├── CodeAlpha_HandwrittenCharacterRecognition.py     # Python training and evaluation pipeline
│   ├── data/
│   │   └── mnist_10k.npz                                # Preprocessed dataset subset
│   └── outputs/
│       └── handwriting_cnn.keras                        # Saved Keras CNN model
└── README.md                                             # Project documentation
```

## 🛠️ Key Features and Model Architecture

- **Dataset:** Uses a preprocessed subset of 10,000 samples stored in `mnist_10k.npz`, containing 28 × 28 grayscale digit images.
- **Convolutional layers:** Extract visual features such as edges, curves, and digit patterns.
- **Pooling layers:** Reduce spatial dimensions while retaining important features.
- **Dropout regularization:** Helps reduce overfitting during training.
- **Softmax output layer:** Produces scores for the 10 digit classes (0–9).
- **Saved model:** Exports the trained network in the native `.keras` format for later evaluation and inference.
- **Training visualization:** The notebook can be used to explore data, training results, and loss and accuracy curves, if implemented in the notebook.

## 🚀 Getting Started

### 1. Prerequisites

Install **Python 3.8 or later**. Then install the required packages:

```bash
pip install numpy tensorflow matplotlib
```

If you plan to run the Jupyter Notebook, install Jupyter as well:

```bash
pip install notebook
```

### 2. Run the Project

Run the commands from the directory containing the outer `CodeAlpha_HandwrittenCharacterRecognition` folder.

#### Option A: Jupyter Notebook

Recommended for exploring the code, data, and training process interactively.

```bash
jupyter notebook CodeAlpha_HandwrittenCharacterRecognition/CodeAlpha_HandwrittenCharacterRecognition.ipynb
```

#### Option B: Python Script

Run the training and evaluation pipeline from the terminal:

```bash
python CodeAlpha_HandwrittenCharacterRecognition/CodeAlpha_HandwrittenCharacterRecognition.py
```

> **Note:** These commands assume the project structure shown above and that the script and notebook use paths relative to the project root. Adjust the paths if your files are arranged differently.

## 💡 Inference Example

Once the model has been trained and saved, you can load it and use it to predict a digit from the dataset.

The example below assumes that `images` is the image-array key in the NPZ file and that the images are stored as 28 × 28 arrays.

```python
import numpy as np
import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model(
    "CodeAlpha_HandwrittenCharacterRecognition/outputs/handwriting_cnn.keras"
)

# Load the dataset
data = np.load(
    "CodeAlpha_HandwrittenCharacterRecognition/data/mnist_10k.npz"
)

# Normalize pixel values to the [0, 1] range
x_test = data["images"].astype("float32") / 255.0

# Prepare one image for the CNN:
# (height, width) -> (batch_size, height, width, channels)
sample_image = np.expand_dims(x_test[0], axis=(0, -1))

# Predict the digit
predictions = model.predict(sample_image, verbose=0)
predicted_digit = int(np.argmax(predictions, axis=1)[0])

print(f"Predicted Digit: {predicted_digit}")
```

**Important:** This example assumes that the dataset contains an `images` array and that the model expects input shaped like `(batch_size, 28, 28, 1)`. Check the keys in your NPZ file and the model's input shape if your dataset or preprocessing pipeline differs. If the dataset contains labels, compare the predicted digit with the corresponding label to evaluate the prediction.

## 📦 Project Artifacts

| File | Description |
|---|---|
| `CodeAlpha_HandwrittenCharacterRecognition.ipynb` | Notebook for interactive development and exploration |
| `CodeAlpha_HandwrittenCharacterRecognition.py` | Python script for running the pipeline |
| `data/mnist_10k.npz` | Preprocessed dataset subset |
| `outputs/handwriting_cnn.keras` | Saved CNN model |

## 🤝 Acknowledgments

- Developed as part of the **CodeAlpha Artificial Intelligence Internship**.
- Uses the **MNIST handwritten digit dataset** as the basis for digit recognition.

---

**Note:** This project focuses on handwritten digit recognition (0–9). Although the repository title mentions characters, recognition of letters or other character classes would require an appropriate dataset and corresponding model training.
