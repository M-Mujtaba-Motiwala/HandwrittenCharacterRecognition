import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED); keras.utils.set_random_seed(SEED)
sns.set_theme(style="white", context="notebook")
print("TensorFlow", tf.__version__)

data = np.load("data/mnist_10k.npz")
images, labels = data["images"], data["labels"].astype(int)

CLASS_NAMES = [str(i) for i in range(10)]
NUM_CLASSES = len(CLASS_NAMES)

print("images:", images.shape, images.dtype, "| value range:", images.min(), "-", images.max())
print("labels:", labels.shape, "| classes:", NUM_CLASSES)
pd.Series(labels).value_counts().sort_index().rename("samples").to_frame().T

fig, axes = plt.subplots(2, 10, figsize=(16, 3.8))
for c in range(NUM_CLASSES):
    idx = np.where(labels == c)[0][:2]
    for r, i in enumerate(idx):
        axes[r, c].imshow(images[i], cmap="gray"); axes[r, c].axis("off")
        if r == 0: axes[r, c].set_title(CLASS_NAMES[c])
plt.suptitle("Two examples per class", y=1.02); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 10, figsize=(16, 2))
for c in range(NUM_CLASSES):
    axes[c].imshow(images[labels == c].mean(axis=0), cmap="magma"); axes[c].axis("off"); axes[c].set_title(CLASS_NAMES[c])
plt.suptitle("Mean image per class", y=1.12); plt.show()

X = (images.astype("float32") / 255.0)[..., np.newaxis]
y = labels

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
print(f"train {X_train.shape[0]} | validation {X_val.shape[0]} | test {X_test.shape[0]}")

import time
t0 = time.time()
svm = SVC(kernel="rbf", C=5, gamma="scale")
svm.fit(X_train.reshape(len(X_train), -1), y_train)
svm_pred = svm.predict(X_test.reshape(len(X_test), -1))
svm_acc = accuracy_score(y_test, svm_pred)
print(f"SVM (RBF) test accuracy: {svm_acc:.4f}   (trained in {time.time() - t0:.0f}s)")

def build_cnn(num_classes=NUM_CLASSES):
    augment = keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.10),
    ], name="augmentation")

    inputs = keras.Input(shape=(28, 28, 1))
    x = augment(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Activation("relu")(x)
        if filters != 128:
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="handwriting_cnn")

model = build_cnn()
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS if "EPOCHS" in globals() else 25,
                    batch_size=128, callbacks=callbacks, verbose=2)

hist = pd.DataFrame(history.history)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
hist[["accuracy", "val_accuracy"]].plot(ax=axes[0], marker="o", ms=3); axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch")
hist[["loss", "val_loss"]].plot(ax=axes[1], marker="o", ms=3); axes[1].set_title("Loss"); axes[1].set_xlabel("epoch")
for a in axes: a.grid(alpha=.3)
plt.tight_layout(); plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
probs = model.predict(X_test, verbose=0)
pred = probs.argmax(axis=1)

print(f"CNN test accuracy: {test_acc:.4f}   |   SVM baseline: {svm_acc:.4f}")
print()
print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=CLASS_NAMES, cmap="Blues", colorbar=False, ax=axes[0])
axes[0].set_title("Confusion matrix (counts)")
cm = confusion_matrix(y_test, pred, normalize="true")
np.fill_diagonal(cm, 0)
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Reds", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=axes[1])
axes[1].set_title("Where the errors are (row-normalised, diagonal removed)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
plt.tight_layout(); plt.show()

wrong = np.where(pred != y_test)[0]
print(f"{len(wrong)} misclassified out of {len(y_test)} test images")
order = wrong[np.argsort(-probs[wrong].max(axis=1))][:18]

fig, axes = plt.subplots(3, 6, figsize=(14, 7.5))
for ax, i in zip(axes.ravel(), order):
    ax.imshow(X_test[i, ..., 0], cmap="gray"); ax.axis("off")
    ax.set_title(f"true {CLASS_NAMES[y_test[i]]} / pred {CLASS_NAMES[pred[i]]}\nconf {probs[i].max():.2f}", fontsize=9, color="#C44E52")
for ax in axes.ravel()[len(order):]: ax.axis("off")
plt.suptitle("Most confident mistakes", y=1.0); plt.tight_layout(); plt.show()

first_conv = [l for l in model.layers if isinstance(l, layers.Conv2D)][0]
w = first_conv.get_weights()[0]
fig, axes = plt.subplots(4, 8, figsize=(10, 5.2))
for k, ax in enumerate(axes.ravel()):
    ax.imshow(w[:, :, 0, k], cmap="coolwarm"); ax.axis("off")
plt.suptitle("32 filters of the first convolutional layer"); plt.tight_layout(); plt.show()

from PIL import Image, ImageDraw, ImageOps

def preprocess_image(img: Image.Image) -> np.ndarray:
    g = ImageOps.grayscale(img)
    a = np.asarray(g, dtype="float32")
    if a.mean() > 127:
        a = 255 - a
    ys, xs = np.where(a > 40)
    if len(xs) == 0:
        raise ValueError("No character found in the image")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(a.shape) * 1.25)
    canvas = np.zeros((side, side), dtype="float32")
    oy, ox = (side - a.shape[0]) // 2, (side - a.shape[1]) // 2
    canvas[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a
    small = Image.fromarray(canvas.astype("uint8")).resize((20, 20), Image.LANCZOS)
    out = np.zeros((28, 28), dtype="float32"); out[4:24, 4:24] = np.asarray(small)
    return out / 255.0

def predict_image(path_or_image, show=True, top_k=3):
    img = Image.open(path_or_image) if isinstance(path_or_image, (str, os.PathLike)) else path_or_image
    x = preprocess_image(img)
    p = model.predict(x[np.newaxis, ..., np.newaxis], verbose=0)[0]
    best = np.argsort(-p)[:top_k]
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].imshow(img, cmap="gray"); ax[0].set_title("input"); ax[0].axis("off")
        ax[1].imshow(x, cmap="gray"); ax[1].set_title(f"model sees -> predicts {CLASS_NAMES[best[0]]}"); ax[1].axis("off")
        plt.show()
    return [(CLASS_NAMES[i], round(float(p[i]), 4)) for i in best]

demo = Image.new("L", (200, 200), 255)
d = ImageDraw.Draw(demo)
d.line([(50, 45), (150, 45)], fill=0, width=14); d.line([(150, 45), (90, 160)], fill=0, width=14)
print(predict_image(demo))

os.makedirs("outputs", exist_ok=True)
model.save("outputs/handwriting_cnn.keras")
print("Saved ->", os.path.abspath("outputs/handwriting_cnn.keras"))
