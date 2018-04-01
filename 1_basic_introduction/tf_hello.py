import tensorflow as tf

#定义一个常量操作节点，
hello = tf.constant("hello, TensorFlow2018!")

#获取一个会话
sess = tf.Session()

#启动会话运行hello节点
print(sess.run(hello))

#任务完成, 关闭会话.
sess.close()