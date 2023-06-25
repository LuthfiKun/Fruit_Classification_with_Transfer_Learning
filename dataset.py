import os
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

def generate(train_data_dir, vaild_data_dir, batch_size = 64):
    #Generating dataset and augmentation for training dataset
    train_generator = ImageDataGenerator(rotation_range=90, 
                                        brightness_range=[0.1, 0.7],
                                        width_shift_range=0.5, 
                                        height_shift_range=0.5,
                                        horizontal_flip=True, 
                                        vertical_flip=True,
                                        preprocessing_function=preprocess_input) # VGG16 preprocessing

    #Generating validation dataset
    valid_generator = ImageDataGenerator(preprocessing_function=preprocess_input) # VGG16 preprocessing

    class_subset = sorted(os.listdir(train_data_dir))

    #Preparing dataset for training
    traingen = train_generator.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    #Preparing dataset for testing
    validgen = valid_generator.flow_from_directory(
        vaild_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return traingen, validgen, len(class_subset)