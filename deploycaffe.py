#加载模块与图像参数设置
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray' 

#模型路径deploy
import caffe
import os
import cv2

#caffe.set_mode_cpu()

model_def =  '/home/xxx/face_detection.prototxt'
model_weights =  '/home/xxx/face_detection.caffemodel'


#模型加载
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  # 那么reshape操作，就是自动将验证图片进行放缩
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
  # transpose将RGB变为BGR,都要做transpose
  # BGR谁放在前面，譬如3*300*100，这里设定3在前面
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
  # 像素点rescale操作，将数据的像素点集中在[0,255]区间内
# transformer.set_mean('data', np.array([104, 117, 123])) 
transformer.set_channel_swap('data', (2,1,0))  

  # CPU classification              
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          300, 300) 
for i in range(0,99):
  imgpath = "/home/xxx/"+("%06d" % i)+".png"
  image = caffe.io.load_image(imgpath) 
  # 导入图片         
  net.blobs['data'].data[...] = transformer.preprocess('data', image)        
  # 预处理图片
  output = net.forward()            
  # 前向传播一次，找出参数
  #net.blobs['data'].data[...] = transformed_image         
  output_prob = output['detection_out']
  det_label = output_prob[0,0,:,1]
  det_conf = output_prob[0,0,:,2]
  det_xmin = output_prob[0,0,:,3]
  det_ymin = output_prob[0,0,:,4]
  det_xmax = output_prob[0,0,:,5]
  det_ymax = output_prob[0,0,:,6]

  # 输出概率
  print 'predicted class is:', det_label
  print 'predicted confidence is:', det_conf
  # 输出最大可能性
  #绘图
  showimg = cv2.imread(imgpath)
  dim = showimg.shape

  for i in range(det_label.size):
    if(det_conf[i]>0.1):
      p1 = (int(dim[1] * det_xmin[i]),int(dim[0] * det_ymin[i])) 
      p2 = (int(dim[1] * det_xmax[i]),int(dim[0] * det_ymax[i]))
      cv2.rectangle(showimg,p1,p2,(0,0,255),2)
  cv2.imshow("1",showimg)
  cv2.waitKey(0)

