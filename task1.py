
"""
Created on Wed Sep 18 19:35:10 2018

@author: MOHSIN AKBAR
"""

from keras.layers import Dense,Convolution2D,MaxPooling2D,Flatten

from keras.models import Sequential



classifier = Sequential()


classifier.add(Convolution2D(16, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))



classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())

classifier.add(Dense(output_dim = 64, activation = 'relu'))
classifier.add(Dense(output_dim = 26, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#  Ftting the CNN 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('english_alphabets/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('english_alphabets/test_set',
                                            target_size = (64, 64),
                                            batch_size = 16,
                                            class_mode = 'categorical')
try:

    classifier.fit_generator(training_set,
                         samples_per_epoch = 200,
                         nb_epoch = 50,
                         validation_data = test_set,
                         nb_val_samples = 35)
except Exception as e:
    print(str(e))
    
# prediction / realization


from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
 

img = image.load_img('h1.jpg', target_size=(64, 64))
var = image.img_to_array(img)
var = np.expand_dims(var, axis=0)
var = preprocess_input(var)

print(classifier.predict(var))



