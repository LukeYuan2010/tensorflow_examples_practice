import tensorflow as tf 

put_into = tf.constant(500., name="put_into")
month_rate = tf.constant(0.05/12, name="month_rate")

pool = tf.Variable(0.0, name="pool")

sum1 = tf.add(put_into, pool, name="sum1")

interest = tf.multiply(sum1, month_rate, name="interest")

sum2 = tf.add(interest, sum1, name="sum2")

# interest_sum = tf.assign()

update_pool = tf.assign(pool, sum2)

init_op = tf.global_variables_initializer()

# merged_summary_op = tf.summary.merge_all()
total_step = 0
with tf.Session() as sess:
	sess.run(init_op)

	writer = tf.summary.FileWriter(logdir='log', graph=sess.graph)
	writer.close()

	for i in range(12*20):
		sum = sess.run(sum2)
		sess.run(update_pool)
		if i%10 == 9:
			print("{}th moth, I have money {} Yuans".format(i, sum))

	