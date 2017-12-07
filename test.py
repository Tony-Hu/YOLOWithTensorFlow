from YOLONN import YoloNeuralNetwork
from VOC2012ImageReader import ImageReader
import tensorflow as tf
from ImageShower import ImageShower
from train import Trainer
import numpy as np
import os

yolo = YoloNeuralNetwork()
weight_files = '/Users/tony/Downloads/YOLO_small.ckpt'
print(weight_files)
sess = tf.Session()
loader = tf.train.Saver()
#loader = tf.train.import_meta_graph(weight_files + str('-201.meta'))
loader.restore(sess, weight_files)

file_name = '2007_000027.jpg'
#input_image = os.path.join(os.path.dirname(os.path.realpath('__file__')),ImageReader.IMAGE_RELATIVE_PATH, file_name)
input_image = '/Users/tony/Google Drive (tony.hu1213@gmail.com)/Project/YOLOWithTensorFlow/VOC2012/JPEGImages/2007_001416.jpg'
image_string = tf.read_file(input_image)
image_decoded = tf.image.decode_jpeg(image_string, channels=3)
image_resized = tf.image.resize_images(image_decoded, ImageReader.RESIZED_PIC_SIZE)
image = tf.cast(image_resized, tf.uint8)
image_resized = sess.run(image_resized)
image = sess.run(image)
net_output = sess.run(yolo.logits,feed_dict={yolo.images:image_resized[np.newaxis]})
print(net_output)
print(net_output.shape)
ImageShower.show_image_and_save(image,net_output)