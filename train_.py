import  numpy as np
import  tensorflow as tf
from yolomodel import  model
s = np.load('vgg16.npy',encoding='latin1')
image = np.random.randint(0,255,(224,224,3))
gt = np.array([ [4,3,2,1,10],[4,2,1,5,15],[2,5,63,6,18],[44,12,144,42,18],[33,44,188,200,8] ])

image = tf.constant(image,dtype=tf.float32)
yolo = model(image,gt,7,224,224,)

optimer = tf.train.GradientDescentOptimizer(0.03).minimize(yolo.total_loss)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):

 #   print("classloss:",sess.run(yolo.Loss['classes']))
  #  print("believeloss:", sess.run(yolo.Loss['believe']))
   # print("coordinateloss:", sess.run(yolo.Loss['coordinate']))
    print('totalloss',sess.run(yolo.total_loss))
    sess.run(optimer)
