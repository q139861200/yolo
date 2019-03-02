import  tensorflow as tf
import  numpy as np

data = np.load('Dataset_utils/data.npy')
ydata =  data[0].reshape(100,1)
xdata = data[1].reshape(100,1)
Saver = tf.train.import_meta_graph('Model/model.ckpt-9601.meta')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess= tf.Session(config=config)
Saver.restore(sess,'Model/model.ckpt-9601')

loss = tf.get_collection('_loss')[0]
x = tf.get_collection('feed_dict')[0]

optimer =  tf.train.GradientDescentOptimizer(0.00001)
grads  = optimer.compute_gradients(loss)
train_op = optimer.apply_gradients(grads)
for i in range(10000):
    sess.run(train_op,feed_dict={x : xdata})
    print('loss:',sess.run(loss,feed_dict={x : xdata}))


