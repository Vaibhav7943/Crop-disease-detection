from tensorflow.keras import layers, models
from keras.utils import image_dataset_from_directory  
from tensorflow.keras.preprocessing.image import ImageDataGenerator



IMG_SIZE = 224  
BATCH_SIZE = 32

val_datagen = ImageDataGenerator(rescale=1./255)    

train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_dir = r'D:\New downloads\dataset\train'
val_dir = r'D:\New downloads\dataset\validation'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)


model.save('plant_disease_model.h5')




