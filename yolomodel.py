import  tensorflow as tf
import  numpy as np
from  tensorflow.python import debug as tf_debug
from  utilities import  *
from  arguments  import  Alpha_coord,Alpha_noobj
from Dataset_utils import  norm_data
class model:
    def __init__(self,image,gt,k,iw,ih):
        self.k = k
        self.iw =float(iw)
        self.ih = float(ih)
        self.gt = gt.astype(np.float32)
        self.gt_number = self.gt.shape[0]
        shape = image.get_shape()

        image = tf.reshape(image,shape=[1,shape[0],shape[1],shape[2]])
        self.image = image
        self.input = self.image
        self.input = self.build_net()
        self.Loss,self.total_loss =  self.construct_loss()
    def build_net(self):
        with tf.variable_scope('conv_1') as scope:
            output= self.get_Shape(self.input)[3]
            w = self.get_variable(scope,3,output)
            mean,var =  tf.nn.moments(self.input,axes=[0,1,2])
            self.input = self.con2d(w,32,1)
            self.input = tf.nn.batch_normalization(self.input,mean,var,0,1.0,0.0001)
            self.input = tf.nn.relu(self.input)
            self.input = self.maxpool()
        with tf.variable_scope('conv_2') as scope:
            output= self.get_Shape(self.input)[3]
            w = self.get_variable(scope,3,output)
            mean,var =  tf.nn.moments(self.input,axes=[0,1,2])
            self.input = self.con2d(w,256,1)
            self.input = tf.nn.batch_normalization(self.input,mean,var,0,1.0,0.0001)
            self.input = tf.nn.relu(self.input)
            self.input =  self.maxpool()
        with tf.variable_scope('conv_3') as scope:
            output= self.get_Shape(self.input)[3]
            w = self.get_variable(scope,3,output)
            mean,var =  tf.nn.moments(self.input,axes=[0,1,2])
            self.input = self.con2d(w,512,1)
            self.input = tf.nn.batch_normalization(self.input,mean,var,0,1.0,0.0001)
            self.input = tf.nn.relu(self.input)
            self.input = self.maxpool()
        with tf.variable_scope('conv_4') as scope:
            output= self.get_Shape(self.input)[3]
            w = self.get_variable(scope,3,output)
            mean,var =  tf.nn.moments(self.input,axes=[0,1,2])
            self.input = self.con2d(w,1024,1)
            self.input = tf.nn.batch_normalization(self.input,mean,var,0,1.0,0.0001)
            self.input = tf.nn.relu(self.input)
            self.input = self.maxpool()
        with tf.variable_scope('conv_5') as scope:
            output= self.get_Shape(self.input)[3]
            w = self.get_variable(scope,3,output)
            mean,var =  tf.nn.moments(self.input,axes=[0,1,2])
            self.input = self.con2d(w,1024,1)
            self.input = tf.nn.batch_normalization(self.input,mean,var,0,1.0,0.0001)
            self.input = tf.nn.relu(self.input)
            self.input = self.maxpool()
        with tf.variable_scope('conv_6') as scope:
            output= self.get_Shape(self.input)[3]
            w = self.get_variable(scope,3,output)
            mean,var =  tf.nn.moments(self.input,axes=[0,1,2])
            self.input = self.con2d(w,1024,1)
            self.input = tf.nn.batch_normalization(self.input,mean,var,0,1.0,0.0001)
            self.input = tf.nn.relu(self.input)
        with tf.variable_scope('fc_1') as scope:
            # shape = self.get_Shape(self.input)
            flat = tf.reshape(self.input,[-1,1])
            output= flat.get_shape()
            w = tf.get_variable(name=scope.name+'01',shape=[4096,output[0]],dtype=tf.float32,initializer=tf.random_uniform_initializer,)
            flat = tf.matmul(w,flat)

        with tf.variable_scope('fc_2') as scope:
            # shape = self.get_Shape(self.input)
            flat = tf.reshape(self.input,[-1,1])
            output= flat.get_shape()
            w = tf.get_variable(name=scope.name+'02',shape=[1470,output[0]],dtype=tf.float32,initializer=tf.random_uniform_initializer,)
            flat = tf.matmul(w,flat)
          #  self.input = tf.nn.sigmoid(flat)
        self.input = tf.reshape(flat,shape=(7,7,30))
        classes = tf.reshape(self.input[:,:,:20],(-1,20))

        believe = tf.reshape(self.input[:,:,20:22],(-1,2))
        boundingbox = tf.reshape(self.input[:,:,22:],(-1,8))
        self.net_package = [classes, believe, boundingbox]
        return self.net_package

    def construct_loss(self):
        Loss={}
        self.index,self.box_valide = anchor_map(self.iw, self.ih, self.gt, self.k) # kk,1    kk,4
        self.box_valide  = self.box_valide.astype(np.int32)
        # index kk 其中含有gt的网格为1 其他为0
        vertical_index = np.where(self.index == 1)[0].astype(np.int32) # 0
        self.column = (np.where(self.box_valide!=0)[0]).astype(np.int32) # 返回的是[0],dtype=int64
        self.valideboundingbox = tf.gather(self.net_package[2],vertical_index,axis=0)
        self.forecast_box = wrap_decode(self.iw,self.ih,self.valideboundingbox) # x,y,w,h(0-1)  ->  x,y,w,h(N,8)
        temp = self.gt[self.column, :]
        print('temp:')
        self.printl = tf.print(self.forecast_box,[self.forecast_box,tf.shape(self.forecast_box)])
        self.forecast_box = tf.cast(tf.reshape(self.forecast_box,(-1,8)),tf.float32)
        print(temp)
        print('======1')
        de_net_coordinate,max_cell_index = wrap_bigger_bb(self.forecast_box,temp)
        gt_box    = encode(self.gt[:,0:4]) # xxyy -> xywh
        Loss["coordinate"] = 1e-10 * Alpha_coord *tf.reduce_sum(tf.square(tf.subtract( de_net_coordinate,gt_box[self.column,:])))
        validebelieve = tf.gather(self.net_package[1],vertical_index)
        # N,2
        iou_believe, iou_distrust  =  wrap_bigger_believe(validebelieve,max_cell_index)
        Loss["believe"] = 1e-3* tf.reduce_sum(tf.square( iou_believe -1)) +Alpha_noobj * tf.reduce_sum(tf.square(iou_distrust-0))
        valide_box = self.gt[:,4]
        valide_box = valide_box[self.column,]
        gt_onehot = tf.one_hot(valide_box,depth=20) #  N,20
        de_net_classes=  tf.gather(self.net_package[0],vertical_index)  # N,20
        print(gt_onehot)
        print(de_net_classes)
        Loss['classes'] = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits(labels=gt_onehot,logits= de_net_classes))
        self.total_loss  =  Loss['classes'] + Loss['believe'] + Loss['coordinate']
        return   Loss,self.total_loss
    def maxpool(self):
            return tf.nn.max_pool(self.input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    def con2d(self,ws,output,stride):
        input_shape = self.get_Shape(self.input)
        self.input =  tf.nn.conv2d(self.input,ws,strides=[1,stride,stride,1],padding='SAME')
        return  self.input

    def get_variable(self,name,kernel,output):
        shape = self.input.get_shape()

        w = tf.get_variable(name='weights',shape=[kernel,kernel, shape[3],output],dtype=tf.float32,
                            initializer = tf.truncated_normal_initializer)
        tf.add_to_collection('l2_loss',tf.nn.l2_loss(w,name=name.name))
        return  w

    def get_Shape(self,M):
        return  M.get_shape()


    #print(sess.run([yolomodel.total_loss,yolomodel.printl]))

'''
data = np.load('Dataset_utils/data.npy')
ydata =  data[0].reshape(100,1)

xdata = data[1].reshape(100,1)
x = tf.placeholder(dtype=tf.float32,shape=[None,1])

w = tf.get_variable(name='w_variable',shape=[100,1],dtype=tf.float32,initializer=tf.random_uniform_initializer)
y = tf.multiply(x,w)
loss = tf.reduce_mean(tf.pow(((y-ydata)),2))
tf.add_to_collection('feed_dict',x)
tf.add_to_collection('_loss',loss)
saver = tf.train.Saver()
optimer = tf.train.GradientDescentOptimizer(0.1)
grads =  optimer.compute_gradients(loss)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i] = [tf.clip_by_value(g,-0.1,0.1),v]
train_op = optimer.apply_gradients(grads)
config =  tf.ConfigProto(allow_soft_placement=True )
config.gpu_options.allow_growth= True
sess=tf.Session(config=config)
sess.run(tf.global_variables_initializer())

#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

for i in range(10000):

    sess.run(train_op,feed_dict={'Placeholder:0':xdata})
    if i % 1000 ==0:
        print('loss:',sess.run(loss,feed_dict={'Placeholder:0':xdata}))
        print('left:',i/10000)
    if i%400 ==0:
        saver.save(sess,'./Model/model.ckpt',global_step=i+1)
'''
