from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
# 使用一个类别为服饰的数据集 Fashion-MNIST(灰度图像)
# 该数据集中，图片的高和宽均为28像素，一共包括了10个类别
# 训练数据集中和测试数据集中的每个类别的图片数分别为60,000和10,000
mnist = input_data.read_data_sets('data/fashion', one_hot=True)  # one_hot是否把标签转为一维向量


# 在一行里画出多张图片
def show_fashion_imgs(images):
    _, figs = plt.subplots(1, len(images), figsize=(15, 15))
    for f, img in zip(figs, images):
        f.imshow(img.reshape((28, 28)))
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 输出正确lables函数
def get_text_labels(labels):
    text_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = []
    for i in np.argmax(labels, 1):
        res.append(text_labels[i])
    return res


# 初始化模型参数
num_inputs = 784
num_outputs = 10
# 定义占位符
X = tf.placeholder(tf.float32, [None, num_inputs])
Y = tf.placeholder(tf.float32, [None, num_outputs])
# 本例中权重和偏差参数分别为784×10和1×10的矩阵
W = tf.Variable(tf.random_normal(shape=(num_inputs, num_outputs), stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))


# 定义softmax运算
def Softmax(x):
    exp = tf.exp(x)
    denominator = tf.reduce_sum(exp, axis=1, keepdims=True)
    return exp / denominator


# 构造模型
y_pred = Softmax(tf.matmul(X, W) + b)
# 交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y_pred), axis=1))
# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 评估模型
# tf.argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))

# 为了确定正确预测项的比例，可以把布尔值转换成浮点数，然后取平均值。
# 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype="float"))

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size=256)
        loss, trainer_ = sess.run([cross_entropy, train_step], feed_dict={X: batch_xs, Y: batch_ys})
        print(loss)

    # 使用模型测试,选取的测试图片可以变
    sample = mnist.test.images[533:552]
    label = mnist.test.labels[533:552]
    print('real labels:', get_text_labels(label))
    pred_label = sess.run(y_pred, feed_dict={X: sample, Y: label}).argmax(axis=1)
    print('predictions:', pred_label)
    # show_fashion_imgs(sample)
    print('accuracy=', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}) * 100, "%")
