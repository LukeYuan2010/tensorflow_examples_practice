import tensorflow as tf 
import numpy
import matplotlib.pyplot as plt 
rng = numpy.random

# 超参数
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# 训练数据
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0] #参数组数

# 计算图的输入节点
X = tf.placeholder('float')
Y = tf.placeholder('float')

# 初始化模型权重
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# 构造一个线性模型
pred = tf.add(tf.multiply(X, W), b)

#均方误差
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
#梯度下降
#注意，minimize()将会改变W和b, 变量的trainable属性默认是True
#这里使用梯度下降法来训练参数，学习率由learning_rate指定
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化变量
init = tf.global_variables_initializer()

#开始训练
with tf.Session() as sess:
    #运行初始化动作
    sess.run(init)

    #载入训练数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict = {X:x, Y:y})

    #每过一定步数（display_step）就显示日志
    if (epoch+1) %display_step == 0:
        c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
        print('Epoch:', '%04d'%(epoch+1), 'cost=', '{:.9f}'.format(c), \
            'W=', sess.run(W), 'b=', sess.run(b), '\n')

    print('最优化训练结束')
    training_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
    print('training_cost=', training_cost, 'W=', sess.run(W), 'b=', sess.run(b), '\n')

    #图表显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W)*train_X+sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    #测试用例
    #应观众要求，增加了测试用例：https://github.com/aymericdamien/TensorFlow-Examples/issues/2
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2))/(2*test_X.shape[0]),
        feed_dict = {X:test_X, Y:test_Y}) #与上面的代价计算使用同样的函数
    print('testing cost=', testing_cost)
    print('Absolute mean square loss difference:', abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W)*train_X+sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()