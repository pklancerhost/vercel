import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load and prepare dataset (Fashion MNIST: 28x28 grayscale images, 10 classes)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1] and add channel dimension (28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 2. Build the CNN model using Keras Sequential API
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    
    # Feature extraction
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Classification head
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()

# 3. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Train the model
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.1
)

# 5. Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# 6. Run inference on a new sample
sample_img = x_test[0:1]  # shape: (1, 28, 28, 1)
actual_label = class_names[y_test[0]]

predictions = model.predict(sample_img)
predicted_class_index = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

print(f"\nPredicted: {class_names[predicted_class_index]} ({confidence:.2f}% confidence)")
print(f"Actual: {actual_label}")