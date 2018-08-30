import time
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import Sequential, activations
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from tensorflow.examples.tutorials.mnist import input_data
# 读取完整的数据集
data_set = input_data.read_data_sets('data/number', one_hot=True)
# 训练集
train_images = data_set.train.images.reshape(55000, 28, 28, 1)
train_labels = data_set.train.labels
# 测试集
test_images = data_set.test.images.reshape(10000, 28, 28, 1)
test_labels = data_set.test.labels

LeNet = Sequential()
# 卷积层块
LeNet.add(Conv2D(filters=6, kernel_size=5, activation=activations.sigmoid))
LeNet.add(MaxPool2D(pool_size=2, strides=2))
LeNet.add(Conv2D(filters=16, kernel_size=5, activation=activations.sigmoid))
LeNet.add(MaxPool2D(pool_size=2, strides=2))
# 全连接层块
LeNet.add(Flatten())
LeNet.add(Dense(120, activation=activations.sigmoid))
LeNet.add(Dense(84, activation=activations.sigmoid))
LeNet.add(Dense(10, activation=activations.softmax))

batch_size = 256
lr = 0.8
num_epochs = 5
LeNet.compile(optimizer=SGD(lr), loss='categorical_crossentropy', metrics=['accuracy'])
for epoch in range(num_epochs):
    start = time.time()
    history = LeNet.fit(train_images, train_labels, batch_size=batch_size, shuffle=True, verbose=0)
    test_res = LeNet.evaluate(test_images, test_labels, verbose=0)
    print('epoch ', epoch + 1, ', loss %.3f' % history.history['loss'][0],
          ', train acc %.9f' % history.history['acc'][0], ', test acc %.9f' % test_res[1],
          ', time %.3f' % (time.time() - start), ' sec')

