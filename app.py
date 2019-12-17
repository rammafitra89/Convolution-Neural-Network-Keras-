#import library
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#inilization CNN
ClassificationEngine = Sequential()
#step 1 - Convolution
ClassificationEngine.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape = (128,128,3), activation = 'relu'))
#step 2 - MaxPolling
ClassificationEngine.add(MaxPooling2D(pool_size =(2,2)))
#add Convolution Layes
ClassificationEngine.add(Conv2D(32,(3,3), activation='relu'))
ClassificationEngine.add(MaxPooling2D(pool_size=(2,2)))
#step 3 - Flattening
ClassificationEngine.add(Flatten())
# Step 4 - Full connection
ClassificationEngine.add(Dense(units = 128, activation = 'relu'))
ClassificationEngine.add(Dense(units = 1, activation = 'sigmoid'))

#step 5 started CNN
ClassificationEngine.compile (optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])
# started traning and test data CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
 
test_datagen = ImageDataGenerator(rescale = 1./255)
 
training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
 
test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')
 
ClassificationEngine.fit_generator(training_set,
                         steps_per_epoch = 8000/32,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = 2000/32)