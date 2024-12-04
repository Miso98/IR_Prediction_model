import tensorflow as tf
# images path
dataset_path ='/home/mitchell/Documents/repos/IR_project/dataset/'


# Load the training dataset with ImageDataGenerator
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + '/train',  # Adjust the directory path
    image_size=(60, 80),  # Match the input shape
    batch_size=32,
    label_mode='binary'  # Assuming binary classification (irregular vs regular)
)

# load the validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + '/val',  # Adjust the directory path
    image_size=(60, 80),
    batch_size=32,
    label_mode='binary'
)

# Create and compile  model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(60, 80, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train 
history = model.fit(train_ds, epochs=20, validation_data=val_ds)

# Evaluate
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc}")


model.save('/home/mitchell/Documents/repos/IR_project/saved_model.h5')

