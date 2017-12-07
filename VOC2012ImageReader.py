import tensorflow as tf
import os
import sys
import numpy as np
'''
Producer: produce images and its labels
'''


class ImageReader:
    if sys.platform == 'win32':
        IMAGE_RELATIVE_PATH = '\\VOC2012\\JPEGImages'
    else:
        IMAGE_RELATIVE_PATH = '/VOC2012/JPEGImages'

    if sys.platform == 'win32':
        LABEL_RELATIVE_PATH = '\\VOC2012\\Annotations'
    else:
        LABEL_RELATIVE_PATH = '/VOC2012/Annotations'

    CURRENT_ABS_PATH = os.path.dirname(os.path.realpath('__file__'))
    RESIZED_PIC_SIZE = [448, 448]
    TRAINING_PERCENT = 0.8
    VALIDATION_PERCENT = 0.1
    TESTING_PERCENT = 0.1
    MAX_OBJECT_BOXES = 56
    MAX_CHAR_SIZE_IN_NUMPY_CHAR_ARRAY = 50
    BATCH_SIZE = 10
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    CELL_SIZE = 7

    @staticmethod
    def __read_image(filename, boxes):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, ImageReader.RESIZED_PIC_SIZE)
        return image_resized, boxes

    @staticmethod
    def read():
        # Get absolute pic paths and label paths as python lists
        pic_names, abs_pic_paths = ImageReader.__get_pic_paths()
        abs_label_paths = ImageReader.__get_label_paths(pic_names)
        training_boxes, validation_boxes, testing_boxes = ImageReader.__get_boxes_from_labels(abs_label_paths)

        # Load all file into tensorflow Dataset
        training_data_set = tf.data.Dataset.from_tensor_slices((abs_pic_paths[0], training_boxes))
        training_data_set = training_data_set.map(ImageReader.__read_image, num_parallel_calls=10)\
            .shuffle(10000).batch(ImageReader.BATCH_SIZE).repeat()

        validation_data_set = tf.data.Dataset.from_tensor_slices((abs_pic_paths[1], validation_boxes))
        validation_data_set = validation_data_set.map(ImageReader.__read_image, num_parallel_calls=10)\
            .shuffle(1000).batch(ImageReader.BATCH_SIZE).repeat()

        testing_data_set = tf.data.Dataset.from_tensor_slices((abs_pic_paths[2], testing_boxes))
        testing_data_set = testing_data_set.map(ImageReader.__read_image, num_parallel_calls=10)\
            .shuffle(1000).batch(ImageReader.BATCH_SIZE).repeat()

        return training_data_set, validation_data_set, testing_data_set

    @staticmethod
    def __get_pic_paths():  # Return pic names(ex: 2007_000027.jpg)
        pic_dir = ImageReader.CURRENT_ABS_PATH + ImageReader.IMAGE_RELATIVE_PATH
        pic_names = os.listdir(pic_dir)
        pic_size = len(pic_names)
        training_size = round(pic_size * ImageReader.TRAINING_PERCENT)
        validation_size = round(pic_size * ImageReader.VALIDATION_PERCENT)
        testing_size = round(pic_size * ImageReader.TESTING_PERCENT)

        relative_train_names = pic_names[0:training_size]
        relative_validation_names = pic_names[training_size:training_size + validation_size]
        relative_testing_names = pic_names[training_size + validation_size:training_size + validation_size + testing_size]
        
        abs_train_paths = ImageReader.__get_abs_pic_paths(relative_train_names)
        abs_validation_paths = ImageReader.__get_abs_pic_paths(relative_validation_names)
        abs_testing_paths = ImageReader.__get_abs_pic_paths(relative_testing_names)
        relative_pic_names = [relative_train_names, relative_validation_names, relative_testing_names]
        abs_pic_paths = [abs_train_paths, abs_validation_paths, abs_testing_paths]
        return relative_pic_names, abs_pic_paths

    @staticmethod
    def __get_abs_pic_paths(relative_pic_paths):
        pic_dir = ImageReader.CURRENT_ABS_PATH + ImageReader.IMAGE_RELATIVE_PATH
        abs_pic_paths = []
        for x in relative_pic_paths:
            full_path = os.path.join(pic_dir, x)
            abs_pic_paths.append(full_path)
        return abs_pic_paths

    @staticmethod
    def __get_label_paths(all_pic_names):
        training_label_dir = ImageReader.CURRENT_ABS_PATH + ImageReader.LABEL_RELATIVE_PATH
        abs_label_paths = []
        for pic_names in all_pic_names:
            abs_label_path = []
            for x in pic_names:
                x = x.replace('jpg', 'xml')
                full_path = os.path.join(training_label_dir, x)
                abs_label_path.append(full_path)
            abs_label_paths.append(abs_label_path)
        return abs_label_paths

    @staticmethod
    def __parse_xmls(labels):
        from xml.dom import minidom
        result_boxes = np.zeros([len(labels), ImageReader.CELL_SIZE, ImageReader.CELL_SIZE,
                                      len(ImageReader.CLASSES) + 5])
        i = 0
        for label in labels:
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
                x_index = int(box[0] * ImageReader.CELL_SIZE / ImageReader.RESIZED_PIC_SIZE[0])
                y_index = int(box[1] * ImageReader.CELL_SIZE / ImageReader.RESIZED_PIC_SIZE[1])
                if result_boxes[i][x_index][y_index][0] == 1:
                    continue
                result_boxes[i][x_index][y_index][0] = 1
                result_boxes[i][x_index][y_index][1:5] = box
                result_boxes[i][x_index][y_index][5 + class_index] = 1
            i += 1
        return result_boxes

    @staticmethod
    def __get_boxes_from_labels(all_labels):
        training_boxes = ImageReader.__parse_xmls(all_labels[0])
        validation_boxes = ImageReader.__parse_xmls(all_labels[1])
        testing_boxes = ImageReader.__parse_xmls(all_labels[2])
        return training_boxes, validation_boxes, testing_boxes

    @staticmethod
    def find_class(name):
        for i in range(len(ImageReader.CLASSES)):
            if ImageReader.CLASSES[i] == name:
                return i
        return -5
