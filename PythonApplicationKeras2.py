from tensorflow.keras.datasets import mnist
(x_train,y_train), (x_test, y_test) = mnist.load_data()

x_train.shape
(6000, 28, 28)
y_train.shape
(6000,)

x_test.shape
(10000, 28, 28)

y_test.shape
(10000,)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the digits

index=np.random.choice(np.arange(len(x_train)), 24, replace=False)
figure,axes=plt.subplots(nrows=4)

for item in zip(axes.ravel(), x_train[index], y_train[index]):
    axes, image, target=item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([]) #remove the x-tick marks
    axes.set_yticks([]) #remove the y-tick marks
    axes.set_title(target)

plt.show()    



# plt.tight_layout()

x_train= x_train #.reshape((10000, 28, 28, 1))
x_train.shape

x_test= x_test #.reshape((60000, 28, 28, 1))
x_test.shape

x_train= x_train.astype('float32')/255
x_test= x_test.astype('float32')/255

from tensorflow.keras.utils import to_categorical

y_train= to_categorical(y_train)
y_train.shape

y_test= to_categorical(y_test)
y_test.shape

