from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224  # You can adjust this based on your model’s input size
BATCH_SIZE = 32

train_dir = r'D:\New downloads\dataset\train'
val_dir = r'D:\New downloads\dataset\validation'

# Image Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the pixel values to [0,1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation data, only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the data from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
