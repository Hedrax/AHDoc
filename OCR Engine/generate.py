import os.path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import math
from config import OCRConfig



def resize_image(image, target_height, max_width, free=False):
    # Calculate the aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    
    # Resize the image to have a height of 64 pixels
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height))

    if (free):
        return resized_image
        
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

def generate_random_polygon(image_shape, num_points=5):
    h, w, _ = image_shape
    points = []
    
    x = np.random.randint(0, w, size=num_points).tolist()
    y = np.random.randint(0, h, size=num_points).tolist()
    
    return np.array([x,y], np.int32).reshape((-1, 1, 2))

def draw_random_drops(image, num_drops, drop_radius, alpha):
    h, w, _ = image.shape
    
    x = np.random.randint(0, w - 1, size=num_drops).tolist()
    y = np.random.randint(0, h - 1, size=num_drops).tolist()
    
    overlay = image.copy()
    
    for i in range(num_drops):
        cv2.circle(overlay, (x[i], y[i]), drop_radius, (0, 0, 0, int(alpha * 255)), -1)
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def generate(cfg, type = 'train', input2= False, valid_aug=False):

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

        #defining max height
        max_height = image.shape[0]
        
        random_idx = np.random.randint(0, dataset_size, size=num_of_words).tolist()
        for i in range(1, num_of_words):
            image2 = cv2.imread(data_dir+image_paths[random_idx[i]])

            #making sure both have same height
            if (max_height > image2.shape[0]):
                image2 = resize_image(image2, max_height, 0, True)
            else:
                max_height = image2.shape[0]
                image = resize_image(image, max_height, 0, True)

            #some white space
            padding_width = 5 
            white_space = np.ones((max_height, padding_width, 3), dtype=np.uint8) * 255

            # Concatenate the images horizontally
            image = np.hstack((image2, white_space, image))
            label += ' ' + labels[random_idx[i]]
        
        
        image = resize_image(image, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)

        #30% augmentend data
        aug_key = (np.random.randint(0, 101) > cfg.CLEAN_DATA_PERCENTAGE)

        #Data Augmentation
        if (valid_aug and aug_key):
            height, width = cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH
            random_list = np.random.randint(0, 2, size=5).tolist()
            #add noise
            if (random_list[0]):
                draw_random_drops(image, cfg.NUM_OF_DROPS, cfg.DROPS_RADIUS, cfg.DROPS_ALPHA)
            
            #draw shapes
            if (random_list[1]):
                polygon_points = generate_random_polygon(image.shape)
                overlay = image.copy()
                
                color = cfg.POLYGON_COLOR  
                alpha = cfg.POLYGON_ALPHA  # Transparency factor
                
                cv2.fillPoly(overlay, [polygon_points], color)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            #rotate with max 15 degrees
            if (random_list[2]):
                
                angle = np.random.uniform(-cfg.MAX_ROTATION_ANGLE, cfg.MAX_ROTATION_ANGLE)
                
                # Calculate the center of the image
                center = (width // 2, height // 2)
                
                # Get the rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Apply the rotation
                image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

            
            #blur
            if (random_list[3]):  
                image = cv2.GaussianBlur(image, (5, 5), 0)

                
            
            #dilute
            if (random_list[3]):
                kernel = np.ones((cfg.DILUTE_KERNAL_FACTOR, cfg.DILUTE_KERNAL_FACTOR), np.uint8)
                image = cv2.dilate(image, kernel, iterations=1)
                
        

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

