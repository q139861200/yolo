import tensorflow as tf
import  numpy as np

def anchor_map(iw,ih,gt,k):
    # iw,ih means the origin image scales
    # gt means gt_boundings
    # k means the last featuremap scales
    bboxNumber =  gt.shape[0]
    center = cal_center(gt,iw,ih)
    #bboxdata= np.hstack((center,gt[:,2:4]))

    coordiate =  np.floor(center*7)
    coordiate =coordiate.astype(np.int)
    coordiate = [i[1]*7+i[0] for i in coordiate] # N, 代表者N个gt大小，每个位置取值0-48
    temp = np.zeros((k*k,))
    temp1 = np.zeros((bboxNumber,))
    for i in range(bboxNumber):
        if temp[coordiate[i]] == 0:
            temp1[i] = 1
            temp[coordiate[i]] = 1
    return unmap(k,coordiate),temp1 #,unmap(k,coordiate,bboxdata)  # kk,1 ,  kk,4,前两个x,y是0-1,w,h是原值
def unmap(k,keep,data = None):
    if data is None :
        res = np.zeros(shape=[k*k,1])
        res[keep,0] = 1
    else:
        wid = data.shape[1]
        res = np.zeros(shape=[k*k,wid])
        res[keep,:] = data
    return  res
def cal_center(gt,iw,ih):
    if gt.shape[1] ==5 :
        locate =  gt[:,:2]
        scale = [ (locate[:,0]/ iw).reshape(-1,1),(locate[:,1] /ih).reshape(-1,1)]
        return  np.hstack(scale)
    else:
        return  0

def decode(iw,ih,origin): #  0<origin <1 to real coordinate

    center_x =  origin[:,[0,4]]*iw
    center_y =  origin[:,[1,5]]*ih
    width =     origin[:,[2,6]]*iw
    height = origin[:,[3,7]]*ih
    res = np.concatenate((center_x, center_y, width, height), axis=1).astype(np.float32)
    return  res

def encode(data):  #x,y,x,y->x,y,w,h
    width = data[:,2] - data[:,0]
    height = data[:,3] - data[:,1]
    center_x =  width/2 + data[:,0]
    center_y = height/2 + data[:,1]
    return   np.stack([center_x,center_y,width,height],axis=1)

def transform_coordinate( data):  #x,y,w,h -> x,y,x,y
    x1 = data[:, 0] - data[:, 2] / 2
    y1 = data[:, 1] - data[:, 3] / 2
    x2 = data[:, 0] + data[:, 2] / 2
    y2 = data[:, 1] + data[:, 3] / 2
    return np.stack( (x1,y1,x2,y2) ).transpose()

def overlaps(anchors,gt): # return value shape [N,]
    N = anchors.shape[0]
    iou = np.zeros((N,),dtype=np.float32 )
    area_1 = (anchors[:,3] -anchors[:,1]) * (anchors[:,2] -anchors[:,0])
    area_2=  (gt[:,3] - gt[:,1]) * (gt[:,2] - gt[:,0])
    for i in range(N):
        iw = min(anchors[i][2],gt[i][2]) -  max(anchors[i][0],gt[i][0])
        if iw>0:
            ih =  min(anchors[i][3],gt[i][3]) - max(anchors[i][3],gt[i][3])
            if ih>0:
                union = iw*ih
                iou[i] = union / (area_1[i]+area_2[i]-union)
    return  iou

def bigger_believe_each_grid(validebelieve,max_cell): # max_cell (N,)
    contrast_ =   1-max_cell
    iou_believe = validebelieve[np.arange(max_cell.shape[0]),max_cell]
    iou_distrust = validebelieve[np.arange(max_cell.shape[0]),contrast_ ]

    return  iou_believe,iou_distrust
def bigger_bb_each_grid(forecast_box,gt):
        box_1 = forecast_box[:,[0,1,2,3]]
        box_2 =  forecast_box[:,[4,5,6,7]]
       # IOU = np.zeros((self.gt.shape[0],2),dtype=np.float32)
        coordinate_1 =  transform_coordinate(box_1) # x,y,w,h to  x1,y1,x2,y2
        coordinate_2 =  transform_coordinate(box_2)
        IOU_0 = overlaps(coordinate_1,gt) # gt  x,y,x,y
        IOU_1  = overlaps(coordinate_2,gt) #  return shape (N,)
        IOU = np.stack( [IOU_0,IOU_1],axis=1 )
        max_cell =   np.argmax(IOU,axis=1).astype(np.int32) # returned shape (N,) tensor
        max_cell_index = np.repeat(np.reshape(max_cell,[-1,1]),4,axis=1)
        max_cell_index = np.where(max_cell_index==0,[0,1,2,3],[4,5,6,7])

        de_net_coordinate=  forecast_box[ np.arange(max_cell_index.shape[0]).reshape((-1,1)),max_cell_index].reshape(-1,4)
        return  de_net_coordinate,max_cell  # (N,4) (N,)N=gt的number
def wrap_bigger_bb(forecast_box,gt):
    return  tf.py_func(bigger_bb_each_grid,[forecast_box,gt],[tf.float32,tf.int32])
def wrap_overlaps(anchors,gt):
    return  tf.py_func(overlaps,[anchors,gt],[tf.float32])
def wrap_bigger_believe(validebelieve,max_cell):
    return tf.py_func(bigger_believe_each_grid,[validebelieve,max_cell],[tf.float32,tf.float32])
def wrap_transform(data):
    return   tf.py_func(transform_coordinate,[data],[tf.float32])
def wrap_decode(iw, ih, origin):
    return tf.py_func(decode, [iw, ih, origin], [tf.float32])
