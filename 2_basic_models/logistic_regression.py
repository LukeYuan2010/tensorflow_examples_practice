
'''
一个使用tensorflow的逻辑回归算法的例子
本例程使用了MNIST手写数据集：
(http://yann.lecun.com/exdb/mnist/)

Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf 

#导入MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=True)

#超参数
learning_rate = 0.02
training_epochs = 25
batch_size = 100
display_step = 1

#tf计算图的输入
x = tf.placeholder(tf.float32, [None, 784]) #mnist中的图片形状是28*28=784
y = tf.placeholder(tf.float32, [None, 10]) #共有0~9十个分类

#模型权重参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#构造模型,前向预测
pred = tf.nn.softmax(tf.matmul(x, W) + b) #soft max

#使用交叉熵最小化误差，构造损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
#使用梯度下降作为优化函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


#初始化变量
init = tf.global_variables_initializer()

#开始训练
with tf.Session() as sess:
    #首先执行初始化的动作
    sess.run(init)

    #循环训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #在batch内的数据上循环训练
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化操作和求代loss
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_xs,
                                                            y:batch_ys})

            #计算loss的平均值
            avg_cost += c/total_batch

        #每个epoch都显示日志信息
        if (epoch+1)%display_step == 0:
            print('Epoch', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
    
    print('Optimization Finished!')

    #评估模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    print('Accuracy:{}'.format(sess.run(accuracy, feed_dict = {x:mnist.test.images, y:mnist.test.labels})))
