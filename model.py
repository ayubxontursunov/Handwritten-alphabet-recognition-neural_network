import os

import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

img = plt.imread('data/img_A3.jpg')
# print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(img.shape)


l1 = 28/img.shape[0]
l2 = 28/img.shape[1]
new_img = np.zeros((28,28))
for x in range(28):
    for y in range(28):
        new_img[x][y] = img[int(x/l1)][int(y/l2)]
new_img = 255 - new_img
# plt.imshow(new_img, cmap ='gray')
# plt.show()
z = np.array(new_img)
z = z.reshape((1,28,28))

data_path = 'my/data/Handwritten_Data.csv'
data = pd.read_csv(data_path, header=None).astype('float32')

X = data.drop([0], axis=1)
y = data[0]
x_train, x_test, train_y, test_y = train_test_split(X, y, test_size=0.2)

x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))


y_train = tf.keras.utils.to_categorical(train_y, num_classes=26, dtype='int')
y_test = tf.keras.utils.to_categorical(test_y, num_classes=26, dtype='int')

print('length of x_train =',len(x_train))
print('length of y_train =',len(y_train))
print()
print('length of x_test =',len(x_test))
print('length of y_test =',len(y_test))
print()

word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


if os.path.exists('../Myproject/keras_model/model_v5'):
    model = tf.keras.models.load_model('../Myproject/keras_model/model_v5')
    print('Loaded from cache')
else:

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(26, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    model.save('../Myproject/keras_model/model_v5')

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


predictions = model.predict([z])
# print(f'true value: {y_train[n]}')
# print(predictions[0])
print(f'predicted value: {word_dict[np.argmax(predictions[0])]}')
plt.imshow(img,cmap ='gray')
plt.text(150, -20, f'predicted value: {word_dict[np.argmax(predictions[0])]}', fontsize=16, color='green')
plt.show()