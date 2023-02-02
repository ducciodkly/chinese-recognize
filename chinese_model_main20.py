import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
# prepare data
nb_classes = 20
batch_size = 128
input_shape = (64, 64, 1)
epoch = 150
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('../input/data-china-20/data_china_20/train', color_mode='grayscale',
                                                 target_size=(64, 64), batch_size=batch_size, class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('../input/data-china-20/data_china_20/test', target_size=(64, 64),
                                            color_mode='grayscale', batch_size=batch_size, class_mode='categorical')
print("Image Processing.......Compleated")

# build model
cnn = tf.keras.models.Sequential()
print("Building Neural Network.....")
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=1024, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=3755, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=nb_classes, activation='softmax'))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Building Neural Network successfull")

print("Training cnn")
cnn.fit(x=training_set, validation_data=test_set, callbacks=[callback], epochs=epoch)
cnn.save('model_20.h5')
print("Saving the bot as model_20.h5")

# evaluated model
score = cnn.evaluate(test_set, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])