from VOC2012ImageReader import ImageReader
from ImageShower import ImageShower
import numpy as np
import tensorflow as tf

def __parse_xmls(labels):
    from xml.dom import minidom
    result_boxes = np.zeros([1, ImageReader.CELL_SIZE, ImageReader.CELL_SIZE,
                             len(ImageReader.CLASSES) + 5])
    print(result_boxes.shape)
    i = 0
    label = labels
    # Starting parse the xml
    xml = minidom.parse(label)
    root = xml.documentElement

    # Get image size (To calculate relative position of the box)
    size = root.getElementsByTagName('size')
    width = size[0].getElementsByTagName('width')[0].firstChild.data
    height = size[0].getElementsByTagName('height')[0].firstChild.data

    # Get boxes and it's classification
    # [[name, relative_x_min, relative_x_max, relative_y_min, relative_y_max]...]
    objects = root.getElementsByTagName('object')
    y_index = 0
    x_index = 0
    for object in objects:
        # Fetch object's name
        name = object.getElementsByTagName('name')[0].firstChild.data

        # Get absolute box position
        bnd_box = object.getElementsByTagName('bndbox')
        x_min = bnd_box[0].getElementsByTagName('xmin')[0].firstChild.data
        y_min = bnd_box[0].getElementsByTagName('ymin')[0].firstChild.data
        x_max = bnd_box[0].getElementsByTagName('xmax')[0].firstChild.data
        y_max = bnd_box[0].getElementsByTagName('ymax')[0].firstChild.data

        # Change to relative path, and store them as a list
        x_min = float(x_min) / float(width)
        x_max = float(x_max) / float(width)
        y_min = float(y_min) / float(height)
        y_max = float(y_max) / float(height)
        box = [(x_max + x_min) / 2, (y_max + y_min) / 2, x_max - x_min, y_max - y_min]
        class_index = ImageReader.find_class(name)
        #x_index = int(box[0] * ImageReader.CELL_SIZE / ImageReader.RESIZED_PIC_SIZE[0])
        #y_index = int(box[1] * ImageReader.CELL_SIZE / ImageReader.RESIZED_PIC_SIZE[1])
        if result_boxes[i][x_index][y_index][0] == 1:
            continue
        result_boxes[i][x_index][y_index][0] = 1
        result_boxes[i][x_index][y_index][1:5] = box
        result_boxes[i][x_index][y_index][5 + class_index] = 1
        y_index += 1
        x_index = y_index // ImageReader.CELL_SIZE
        y_index = y_index % ImageReader.CELL_SIZE
        print(result_boxes[i][x_index][y_index][1:5])
    i += 1
    return result_boxes


#filename = 'C:\\Users\\Azrael\\Google_Drive\\Project\\VOC2012\\JPEGImages\\2007_000027.jpg'
filename = '/Users/tony/Google Drive (tony.hu1213@gmail.com)/Project/YOLOWithTensorFlow/VOC2012/JPEGImages/2008_000123.jpg'
image_string = tf.read_file(filename)
image_decoded = tf.image.decode_jpeg(image_string, channels=3)
image_resized = tf.cast(tf.image.resize_images(image_decoded, ImageReader.RESIZED_PIC_SIZE),tf.uint8)
sess = tf.Session()
image = sess.run(image_resized)
box = __parse_xmls('/Users/tony/Google Drive (tony.hu1213@gmail.com)/Project/YOLOWithTensorFlow/VOC2012/Annotations//2008_000123.xml')
ImageShower.show_image_and_save(image, box[0])

print(box)