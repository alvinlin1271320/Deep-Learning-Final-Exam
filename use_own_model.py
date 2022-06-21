from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

src_path_train = r"./train/"

train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.10)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

batch_size = 8
train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(500, 500),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="binary",
    subset='training',
    shuffle=True,
    seed=42,
)

print(train_generator.samples)

valid_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(500, 500),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="binary",
    subset='validation',
    shuffle=True,
    seed=42
)

print(valid_generator.samples)

def prepare_model():
    model = Sequential()
    model.add(Conv2D(64,kernel_size=(5,5),activation='relu',input_shape=(500, 500, 3)))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model

model = prepare_model()
model.fit(train_generator, validation_data = train_generator, 
          steps_per_epoch = train_generator.n//train_generator.batch_size,
          validation_steps = valid_generator.n//valid_generator.batch_size,
          epochs=30)
model.save_weights('./my_model_weight.h5')