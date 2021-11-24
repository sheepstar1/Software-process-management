import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


#导入数据集
fashion_mnist = keras.datasets.fashion_mnist
#train_images&train_lablels是训练集，而test_images&test_labels是测试集
(train_images, train_labels), (test_images, test_lables) = fashion_mnist.load_data()

#数据集标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Shape of the train data")
print("train_images",train_images.shape)
print()
print("train_lables",len(train_labels))
print()

print("Shape of the test")
print("test_images",test_images.shape)
print()
print("test_labels",len(test_lables))
print()

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255
test_images = test_images /255

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_lables,verbose=2)
print("\nTest accuracy:",test_acc)
print()

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

prediction = probability_model.predict(test_images)

print(prediction[0])

np.argmax(prediction[0])

test_lables[0]


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, prediction[i], test_lables, test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction[i],  test_lables)
plt.show()


n = 20
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(n,prediction[n], test_lables, test_images)
plt.subplot(1,2,2)
plot_value_array(n, prediction[n], test_lables)
plt.show()

nuw_rows = 5
nuw_cols = 3
num_images = nuw_rows*nuw_cols
plt.figure(figsize=(2*2*nuw_cols,2*nuw_rows))

for i in range(num_images):
    plt.subplot(nuw_rows, 2*nuw_cols, 2*i+1)
    plot_image(i, prediction[i], test_lables, test_images)
    plt.subplot(nuw_rows, nuw_cols*2, 2*i+2)
    plot_value_array(i, prediction[i], test_lables)
plt.tight_layout
plt.show()