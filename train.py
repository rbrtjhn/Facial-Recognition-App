import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models

# 1. Load the dataset directly from your folders
dataset_path = "Data" # Make sure this matches your main folder name!
batch_size = 32
img_height = 224 
img_width = 224

train_ds = image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = image_dataset_from_directory(
  dataset_path,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# 2. Get class names automatically!
class_names = train_ds.class_names
print(f"Found Classes: {class_names}")

# Save the labels to a file so app.py can read them
with open('labels.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

# 3. Build a fast, lightweight AI (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze base model for speed

model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(224, 224, 3)),
  base_model,
  layers.GlobalAveragePooling2D(),
  layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train the model! (Only 5 epochs for speed)
print("Training started... this will take a few minutes!")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 5. Save the final file!
model.save('keras_model.h5')
print("Model saved! You are ready to run app.py!")