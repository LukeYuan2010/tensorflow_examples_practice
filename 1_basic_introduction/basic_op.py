import tensorflow as tf 

#定义两个常量操作
#构造函数返回的值就是常量节点(Constant op)的输出
a = tf.constant(2)
b = tf.constant(3)

#启动默认的计算图
with tf.Session() as sess:
	print("a = 2, b = 3")
	print("常量相加：{}".format(sess.run(a+b)))
	print("常量相乘：{}".format(sess.run(a*b)))

#使用变量输出值作为计算图的输入
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

#定义一些操作
add = tf.add(a, b)
mul = tf.multiply(a, b)

#启动默认的计算图
with tf.Session() as sess:
	print("变量相加：{}".format(sess.run(add, feed_dict={a:2, b:3})))
	print("变量相乘：{}".format(sess.run(mul, feed_dict={a:2, b:3})))


#创建一个1X2的常量矩阵，该op会作为一个节点被加入到默认的计算图
#构造器返回的值代表这个op的输出
matrix1 = tf.constant([[3., 3.]])

#创建一个2X1的常量矩阵
matrix2 = tf.constant([[2.], [2.]])

#创建一个矩阵乘法op，它的输入为matrix1和matrix2
#返回的值product表示乘法的结果
product = tf.matmul(matrix1, matrix2)

#为了运行mutmul op我们运行会话的run()方法，使用product作为输入，product代表mutmul op的输出
#这表明我们想要matmul op的输出

#op的所有输入都会由会话自动运行。这些输入一般都是并行运行的

#对'run(product)'的调用回引起这3个op的执行：2个constants和一个matmul
#op的输出值返回给result，这是一个numpy数组对象
with tf.Session() as sess:
	result = sess.run(product)
	print("矩阵常量相称：{}".format(result))
