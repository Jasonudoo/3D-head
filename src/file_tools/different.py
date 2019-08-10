#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io,transform
import glob
import os
import time
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

inputs = '../../data1/train/'
outputs='../../data1/tfrecord-3/'
#num_class = ['c0','c1']
img_size=[224,224]
overlap = 3
#a=overlap


def create_record():
    '''
    0 -- img1.jpg
         img2.jpg
         img3.jpg
         ...
    1 -- img1.jpg
         img2.jpg
         ...
    2 -- ...
    ...
    '''
    maxrecordnum = 80
    recordfilenum = 0
    recordnum=0
    #writer = tf.python_io.TFRecordWriter(outputs+"train.tfrecords")
    #ftrecordfilename = ("train.tfrecords-%.3d" % recordfilenum)
    writer = tf.python_io.TFRecordWriter(outputs+"train.tfrecords-%.3d" % recordfilenum)
    cate=[inputs+x for x in os.listdir(inputs) if os.path.isdir(inputs+x)]
    print(cate)
    for index, name in enumerate(cate):
        class_path = name + "/"
        img_rawover = ''
        m=1

        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize([224,224])


            img_raw = img.tobytes() #bytes
            #print("befor",img_raw1)
            img_rawover=img_rawover+img_raw
            #print("1")
            #print("after",img_rawover)
            m+=1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            if(m > overlap):
                example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_rawover]))
                })) 
                m=1
                img_rawover = ''
            
            #writer = tf.python_io.TFRecordWriter(outputs+"train.tfrecords-%.3d" % recordfilenum) 
                writer.write(example.SerializeToString())

            #img_tower=img_tower +img_tower
            recordnum+=1
            if(recordnum >= maxrecordnum):
                recordfilenum +=1
                recordnum=0
                writer.close()
                #writer = tf.python_io.TFRecordWriter(outputs+"train.tfrecords-%d-%.3d" %(random.randint(0, 200),recordfilenum))           
                writer = tf.python_io.TFRecordWriter(outputs+"train.tfrecords-%.3d" %(recordfilenum))
    writer.close()
    #print(img_tower)
def read_and_decode(data_path):
    data_files = tf.gfile.Glob(data_path+'*')
    #filename_queue = tf.train.string_input_producer([data_files])
    #tf.RandomShuffleQueue()
    #filename_queue = tf.train.string_input_producer(tf.random_shuffle(data_files),shuffle=True)
    filename_queue = tf.train.string_input_producer(data_files,shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
        
    
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    #print(img)

    img = tf.reshape(img, [overlap,224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    
    return img, label

if __name__ == '__main__':
    create_record()
    
    img, label = read_and_decode(outputs)
 
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=1, capacity=2000,
                                                    min_after_dequeue=1000,
                                                    num_threads=5)
    #op
    init = tf.global_variables_initializer()
 
    with tf.Session() as sess:
        sess.run(init)
    
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(1):
            val, l= sess.run([img_batch, label_batch])
            #l = to_categorical(l, 12)
            print(val.shape, overlap)
            print(val)
            print(i)
        
          


