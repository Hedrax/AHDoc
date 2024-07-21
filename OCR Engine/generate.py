import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup



def resize_image(image, target_height, max_width):
    # Calculate the aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    
    # Resize the image to have a height of 64 pixels
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height))
    
    # If the width exceeds the max width, scale down to max width
    if new_width > max_width:
        resized_image = cv2.resize(resized_image, (max_width, target_height))
    else:
        # Otherwise, pad the image with white space to the right
        padding_width = max_width - new_width
        white_space = np.ones((target_height, padding_width, 3), dtype=np.uint8) * 255
        resized_image = np.hstack((resized_image, white_space))
    
    return resized_image

def vectorize_label(label, char_to_num, cfg):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = cfg.MAX_LEN - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=cfg.PADDING_TOKEN)
    return label


def generate(cfg, type = 'train', input2= False):

    match type:
        case 'train':
             data_dir = cfg.TRAIN_DATA_PATH
        case 'val':
             data_dir = cfg.VALIDATION_DATA_PATH
        case _:
            data_dir = cfg.TEST_DATA_PATH

    #without the directory
    image_paths = []
    labels = []

    char_to_num = StringLookup(vocabulary=cfg.characters, mask_token=None)
    
    with open(osp.join(data_dir, 'labels.txt'), encoding="utf-8") as f:
         #format :: 1_gtLabel
        read = f.readlines()

        for i in range (len(read)):
            #find the _
            split = 1
            while(True):
                if (read[i][split] == '_'):
                    break;
                split+=1
            image_paths.append(str(read[i][0:split])+'.png')
            labels.append(read[i][split+1:-1])

    dataset_size = len(image_paths)
    #array of numbers from 0 to dataset_size
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    def init_input():
        batch_images = np.zeros([cfg.BATCH_SIZE, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3], dtype=np.float32)
        batch_labels = np.zeros((cfg.BATCH_SIZE, cfg.MAX_LEN), dtype=np.int32)
        return batch_images, batch_labels
    
    current_idx = 0
    b = 0

    while True:
        if current_idx >= dataset_size:
            np.random.shuffle(indices)
            current_idx = 0
        if b == 0:
            batch_images, batch_labels = init_input()

        #choosing between 1 and max number -1 to be added
        num_of_words = np.random.randint(1, cfg.MAX_NUMBER_OF_WORDS)
        
        #concate words of the upper defined indices and print out the true labels of them
        #get the images of those indices and concatenate with each other along with labels 



        # Read the images
        image = cv2.imread(data_dir+image_paths[current_idx])
        label = labels[current_idx]
        for i in range(1, num_of_words):
            random_idx = np.random.randint(1, dataset_size)
            image2 = cv2.imread(data_dir+image_paths[random_idx])

            #some white space
            padding_width = 5 
            white_space = np.ones((image.shape[0], padding_width, 3), dtype=np.uint8) * 255

            # Concatenate the images horizontally
            image = np.hstack((image2, white_space, image))
            label += ' ' + labels[random_idx]
        
        
        image = resize_image(image, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)

        #normalize image
        image = tf.cast(image, np.float32) / 255.0


        label = vectorize_label(label, char_to_num,cfg)


        
        batch_images[b] = image
        batch_labels[b] = label
        b += 1
        current_idx += 1


        if b == cfg.BATCH_SIZE:
            if (input2):
                yield [batch_images, batch_labels], batch_labels
            else:
                yield batch_images, batch_labels
            b = 0

