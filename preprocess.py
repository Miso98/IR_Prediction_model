import tensorflow as tf
import matplotlib.pyplot as plt

# Directories
train_dir = '/home/mitchell/Documents/repos/IR_project/dataset/train'
val_dir = '/home/mitchell/Documents/repos/IR_project/dataset/val'
test_dir = '/home/mitchell/Documents/repos/IR_project/dataset/test'

# Load the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(60, 80),  # Resize images to match the IR image dimensions
    batch_size=32,  # You can adjust the batch size as needed
    label_mode='binary'  # Use binary labels (0 for regular, 1 for irregular)
)

# view the dataset's class names
class_names = train_ds.class_names
print("Class names:", class_names)

# Apply rescaling using a map operation
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))  # Normalize pixel values to [0, 1]

# Inspect the first batch of images and labels
for images, labels in train_ds.take(1):
    print(images.shape)  # Should print (batch_size, 60, 80, 3)
    print(labels)

    #plot  images
    plt.figure(figsize=(10, 10))
    for i in range(9):  # Display 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])  # Display the image
        # Convert label tensor to numpy array and use as index
        plt.title(class_names[int(labels[i].numpy())])  # Convert label to integer and display the class name
        plt.axis("off")
    plt.show()






