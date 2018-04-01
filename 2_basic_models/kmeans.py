import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# 本代码演示K均值的用法， tensorflow版本必须大于等于V1.1.0
# 代码项目：Project: https://github.com/aymericdamien/TensorFlow-Examples/

# 由于tensorflow实现的K均值算法无法从GPU中获得额外好处，所以我们忽略GPU设备
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 导入MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data', one_hot=True)
full_data_x = mnist.train.images # shape:(55000, 784)，注意记住这个55000，理解后面会用到

# 模型超参数
num_steps = 50  # 训练的总步数
batch_size = 1024  # 每个batch的样本数
k = 5000  # K的大小
num_classes = 10  # 十个数字，这也是模型最终分类的个数
num_features = 784  # 每个图片都是28X28，共784个像素

# 输入图片
X = tf.placeholder(tf.float32, shape=[None, num_features])
# 标注
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means的参数，其实是从库里使用提前封装好的图
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

# 构建K-Means的计算图
training_graph = kmeans.training_graph()

if len(training_graph) > 6:  # tensorflow 1.4及以上版本
    (all_scores, cluster_idx, scores, cluster_cnters_initialized,
     cluster_cnters_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_cnters_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0] # 存放所有数据的图心序号
avg_distance = tf.reduce_mean(scores) # 存放平均距离

# 初始化变量
init_vars = tf.global_variables_initializer()

# 建立一个tensorflow会话
sess = tf.Session()

# 运行初始化操作
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# 训练
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X: full_data_x})

    if i % 10 == 0 or i == 1:
        print('步骤 %i， 平均距离是：%f' % (i, d))

# 给每个图心分配一个标签
# 计算每个图心的样本个数，把样本归入离它最近的图心（使用idx）
counts = np.zeros(shape=(k, num_classes))  # counts的shape是(25, 10),用于存放25个图心分类的频率计数
for i in range(len(idx)):
    # idx的shape是(55000,),每个成员都是0~24之间的值，对应所属图心的编号
    counts[idx[i]] += mnist.train.labels[i]
    # mnist.train.labels的shape是(55000, 10), 每个成员都是独热编码，用来标注属于哪个数字

# 将最高频的标注分配给图心。 len(labels_map)是25,也就是每个图心一个成员，记录每个图心所属的数字分类
labels_map = [np.argmax(c) for c in counts]
# 转换前，labels_map的shape为(25,)
labels_map = tf.convert_to_tensor(labels_map)
# 此时labels_map变成了一个const op，输出就是上面(25,)包含的值

# 评估模型。下面开始构建评估计算图
# 注意：centroid_id就是对应label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# cluster_idx输出的tensor，每个成员都映射到labels_map的一个值。
# cluster_label的输出就是映射的label值，后面用来跟标注比较计算准确度

# 计算准确率
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 测试模型
test_x, test_y = mnist.test.images, mnist.test.labels
print("测试准确率：", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
