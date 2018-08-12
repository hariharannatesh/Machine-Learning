import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
learning_rate=0.001
epochs=10
batch_size=100
X=tf.placeholder(tf.float32,[None,784],name='X')
y=tf.placeholder(tf.float32,[None,10],name='y')
W1=tf.Variable(tf.random_normal([784,300],stddev=0.3),name='W1')
b1=tf.Variable(tf.random_normal([300]),name='b1')
W2=tf.Variable(tf.random_normal([300,10],stddev=0.3),name='W2')
b2=tf.Variable(tf.random_normal([10]),name='b2')
hidden_out=tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
y_=tf.nn.softmax(tf.add(tf.matmul(hidden_out,W2),b2))
#y_clipped=tf.clip_by_value(y_,1e-10,0.99999999)
#cross_entropy=-tf.reduce_mean(tf.reduce_sum(y*tf.log(y_clipped)+(1-y)*tf.log(1-y_clipped)),axis=1)
#optimiser=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=y))
optimizer=tf.train.AdamOptimizer(0.0001).minimize(cost)
init_op=tf.global_variables_initializer()
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    sess.run(init_op)
    total_batch=int(len(mnist.train.labels)/batch_size)
    for epoch in range(epochs):
        avg_cost=0
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_x,y:batch_y})
            avg_cost+=c
        print("Epoch:",(epoch+1),"cost=","{:.3f}".format(avg_cost))
    print(sess.run(accuracy,feed_dict={X:mnist.test.images,y:mnist.test.labels}))
