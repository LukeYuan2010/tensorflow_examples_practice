import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
multiplication = tf.multiply(input1, intermed)

sess = tf.Session()

result = sess.run([multiplication, intermed])
print(result)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))


input1 = tf.constant(3.0)
input2 = tf.constant(5.0)
output2 = tf.multiply(input1, input2)

print(sess.run([output2], feed_dict={input1:7., input2:2.}))
print(sess.run([output2]))
sess.close()