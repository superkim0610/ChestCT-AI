from tensorflow.keras.preprocessing import image
from tensorflow.image import resize, pad_to_bounding_box
from tensorflow.keras import layers, models
import numpy as np
# from tensorflow.keras.models import load_weights
import time

def img_to_array(img):
    target_size = 224

    img = img.convert('L')
    img = image.img_to_array(img)
    img = img / 255

    # image resize
    ratio = target_size / max(img.shape)
    img = resize(img, (int(img.shape[0] * ratio), int(img.shape[1] * ratio)))

    # image padding
    img = pad_to_bounding_box(img, int((target_size-img.shape[0])/2), int((target_size-img.shape[1])/2), target_size, target_size)
    
    return img

def array_standardize(array):
    import pickle

    with open("mean_and_std.pickle", 'rb') as f:
        data = pickle.load(f)
        mean_vals = data['mean_vals']
        std_val = data['std_val']
        
    std_array = (array - mean_vals) / std_val # check
    return std_array
    
# model structure
def create_model():
  model = models.Sequential()

  # feature extractor
  # block1
  model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 1), name='block1_conv1'))
  model.add(layers.MaxPool2D((2, 2), name='block1_pool'))

  # block2
  model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(112, 112, 1), name='block2_conv1'))
  model.add(layers.MaxPool2D((2, 2), name='block2_pool'))

  # block4
  model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(56, 56, 1), name='block3_conv1'))
  model.add(layers.MaxPool2D((2, 2), name='block3_pool'))

  # block5
  model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1), name='block4_conv1'))
  model.add(layers.MaxPool2D((2, 2), name='block4_pool'))

  # block6
  model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu', input_shape=(14, 14, 1), name='block5_conv1'))
  model.add(layers.MaxPool2D((2, 2), name='block5_pool'))

  # classfier
  model.add(layers.AveragePooling2D((7, 7)))


  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(128, activation='relu'))

  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(128, activation='relu'))

  model.add(layers.Dense(4, activation='softmax'))
  model.add(layers.Flatten())

  return model

def predict_img(img):
    array = img_to_array(img)
    std_array = array_standardize(array)
    # print(std_array.shape)
    
    t0 = time.time()
    
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.load_weights("m23(224).keras") # check
    
    t1 = time.time()
    
    result = model.predict(np.array([std_array]))
    # print(result)
    print(f"{t1-t0}s consumed")
    
    return result