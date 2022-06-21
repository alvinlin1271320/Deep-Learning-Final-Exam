from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

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

img = image.load_img("./sunflowers_1.jpg",target_size=(500,500))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
saved_model = prepare_model()
saved_model.load_weights("./my_model_weight.h5")
output = saved_model.predict(img)
print("daisy: ", output[0][0])
print("dandelion: ", output[0][1])
print("roses: ", output[0][2])
print("sunflowers: ", output[0][3])
print("tulips: ", output[0][4])