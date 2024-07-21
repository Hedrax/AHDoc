#Written by Alhossien.waly@ejust.edu.eg

import cv2
import numpy as np
import os

def read_yolo_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual values
            values = line.strip().split()
            
            values = values[1:] + [values[0]]
            # Convert values to floats
            values = [float(val) for val in values]
            # Append to labels list
            labels.append(values)
    return labels

def read_labels(file_path):
    
    coordinates = []

    # Open the file
    with open(file_path, 'r') as file:
        # Read each line
        lines = file.readlines()

    arrays = []
    for line in lines:
        pairs = (line[:-3]).strip().split(',')
        coordinates = [(int(pairs[i]), int(pairs[i+1])) for i in range(0, len(pairs), 2)]
        arrays.append(coordinates)
    return arrays

def yolo_to_polygon(yolo_coords, image_height, image_width):
    x, y, w, h = map(float, yolo_coords)
    x *= image_width
    w *= image_width
    y *= image_height
    h *= image_height
    x_min = round(x - (w / 2))
    y_min = round(y - (h / 2))
    x_max = round(x + (w / 2))
    y_max = round(y + (h / 2))
    segment = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    return segment


def yolo_to_abs(labels, image_height, image_width):
    temp_array = []
    for label in labels:
        temp = yolo_to_polygon(label[:4], image_height, image_width)
        #it represents label 
        temp.append(0)
        temp_array.append(temp)
    return temp_array


def draw_on_image(image,labels):
    img = image.copy()
    image = None
    boxes = []
    for label in labels:
        boxes.append(label[:4])
    for box in boxes:
        cv2.drawContours(img, [np.array(box)], -1, (0, 255, 0), 2)
    return img



def get_image_corresponding_labels(image_dir,text_dir, JPGADD = False):

    # Get a list of all image files
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    # Get corresponding text files based on image filenames
    text_files = []
    for image_file in image_files:
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(image_file))[0]
        extension = os.path.splitext(os.path.basename(image_file))[1]
        
        if (JPGADD):
            filename+= extension
        # Assuming text files have the same name as image files but with a .txt extension
        text_file = os.path.join(text_dir, filename + '.txt')

        # Check if the corresponding text file exists
        if os.path.exists(text_file):
            text_files.append(text_file)
        else:
            text_files.append(None)  # If no corresponding text file found
            
    return image_files,text_files

def load_imgs_labels(image_files,text_files, YOLO=True):
    # Load images and corresponding text data
    images = []
    texts = []
    for i in range (len(image_files)):
        # Load image
        image = cv2.imread(image_files[i])
        if image is not None:
            images.append(image)
            # Load text data if corresponding text file exists
            if (YOLO):
                texts.append(read_yolo_labels(text_files[i]))
            else:
                
                print(text_files[i])
                texts.append(read_labels(text_files[i]))
    return images, texts
    
def list_conversion_yolo_to_abs(images, texts):
    temp = []
    for i in range(len(images)):
        temp.append(yolo_to_abs(texts[i], images[i].shape[0],images[i].shape[1]))
    return temp

def draw_list_images(images, texts):
    drawed_images = []
    for i in range(len(images)):
        drawed_images.append(draw_on_image(images[i], texts[i]))
    return drawed_images

#########################################################################################
#########################################################################################
#########################################################################################

def visualize_ground_truth(image_dir,text_dir, output_path, YOLO_FORMAT= False, StdFormat = True):
    # Get a list of all image files
    image_files, text_files = get_image_corresponding_labels(image_dir,text_dir,True)
    size = len(image_files)
    
    print(len(text_files) , " labels files extracted")
    # Load images and corresponding text data
    for i in range(20, size, 20):
        images, texts = load_imgs_labels(image_files[i-20:i],text_files[i-20:i],YOLO_FORMAT)
    
        if (YOLO_FORMAT):
            temp = list_conversion_yolo_to_abs(images, texts)
        
        #draw on images
        drawed_images = draw_list_images(images, texts)
        for j in range(len(drawed_images)):
            cv2.imwrite(output_path+str(i-20+j)+'.jpg', drawed_images[j])



#that function is useful for YOLO FORMATED
def output_list_imgs_labels(images, texts, image_out_dir, label_out_dir):
    for i in range (1,len(images)+1):
        # Load image
        image = images[i-1]
        if image is not None:
            # Save image with incremented filename
            image_output_path = os.path.join(image_out_dir, f'{i}.jpg')
            cv2.imwrite(image_output_path, image)
    
            # Save text with incremented filename
            temp = yolo_to_abs(texts[i-1], images[i-1].shape[0],images[i-1].shape[1])
            output_string = ""
            
            text_output_path = os.path.join(label_out_dir, f'{i}.txt')
            for j in range(len(temp)):
                data = temp[j][:-1]
                # Flatten the list and format coordinates
                formatted_data = [str(coord) for point in data for coord in point]
                # Join the coordinates with commas
                output_string += ",".join(formatted_data)
                output_string += ","+str(0)
                if (i != len(temp)-1):
                    output_string += '\n'
            with open(text_output_path, 'w') as text_output_file:
                text_output_file.write(output_string)



#spliting functions
def read_split_write(input_dir, out_dir):
    img_input = input_dir+"images/"
    lbl_input = input_dir+"labels/"

    img_out = out_dir+"images/"
    lbl_out = out_dir+"labels/"
    
    # Get a list of all image files
    image_files, text_files = get_image_corresponding_labels(img_input,lbl_input)
    
    # Load images and corresponding text data
    images, texts = load_imgs_labels(image_files,text_files, False)
    
    #doing a 1/4 split
    sub_imgs, sub_lbls = process_images_with_labels(images, texts)
    #doing a 1/3 split
    sub_imgs2, sub_lbls2 = process_images_with_labels(images, texts,3)
    #doing a 1/2 split
    # sub_imgs3, sub_lbls3 = process_images_with_labels(images, texts,2)
    
    #appending into one array
    sub_imgs.extend(sub_imgs2)
    sub_lbls.extend(sub_lbls2)

    # sub_imgs.extend(sub_imgs3)
    # sub_lbls.extend(sub_lbls3)
    
    write_images_labels(sub_imgs, sub_lbls, img_out, lbl_out)
    

def write_images_labels(images, labels_list, img_out, lbl_out):
    # Output processed images and labels
    for i, (image, labels_list) in enumerate(zip(images, labels_list)):
        # Write image
        image_path = os.path.join(img_out, f'{i+1}.jpg')
        cv2.imwrite(image_path, image)
    
        # Write labels
        label_path = os.path.join(lbl_out, f'{i+1}.jpg.txt')
        output_string = ""
        for j in range(len(labels_list)):
            data = labels_list[j]
            # Flatten the list and format coordinates
            formatted_data = [str(coord) for point in data for coord in point]
            # Join the coordinates with commas
            output_string += ",".join(formatted_data)
            output_string += ","+str(0)
            if (j != len(labels_list)-1):
                output_string += '\n'
        with open(label_path, 'w') as text_output_file:
            text_output_file.write(output_string)

def process_images_with_labels(images, label_lists,split_factor= 4):
    processed_images = []
    processed_labels = []

    for i in range(len(images)):
        # Read the image
        image = images[i]
        
        # Get the labels for this image
        labels = label_lists[i]

        # Split the image into sub-images with adjusted labels
        sub_images, sub_labels = split_image(image, labels,split_factor=split_factor)

        # Append the sub-images and sub-labels to the processed lists
        processed_images.extend(sub_images)
        processed_labels.extend(sub_labels)

    return processed_images, processed_labels


def split_image(image, labels, num_splits=16, overlap_ratio=0.4, split_factor= 4):
    # Calculate the dimensions of each sub-image
    height, width = image.shape[:2]
    split_height = height // split_factor
    split_width = width // split_factor
    
    # Calculate overlap pixel count
    overlap_height = int(split_height * overlap_ratio)
    overlap_width = int(split_width * overlap_ratio)
    
    sub_images = []
    sub_labels = []
    
    for row in range(split_factor):
        for col in range(split_factor):
            # Calculate the starting and ending coordinates for cropping
            start_y = row * (split_height - overlap_height)
            end_y = min(start_y + split_height, height)
            start_x = col * (split_width - overlap_width)
            end_x = min(start_x + split_width, width)
            
            # Crop the sub-image
            sub_image = image[start_y:end_y, start_x:end_x]
            
            # Adjust labels for the sub-image
            adjusted_labels = []
            for label in labels:
                adjusted_label = [(max(min(point[0] - start_x, split_width), 0), max(min(point[1] - start_y, split_height), 0)) for point in label]
                # Check if the label intersects with the sub-image
                if any(0 <= point[0] < split_width and 0 <= point[1] < split_height for point in adjusted_label):
                    adjusted_labels.append(adjusted_label)
            
            # Drop labels with visible width <= 0.4 of the original label width
            # Drop labels with visible height < 0.8 of the original label height
            adjusted_labels = [label for label in adjusted_labels if 
                               max(label, key=lambda x: x[0])[0] - min(label, key=lambda x: x[0])[0] >= 0.3 * (max(labels[0], key=lambda x: x[0])[0] - min(labels[0], key=lambda x: x[0])[0]) and
                               max(label, key=lambda x: x[1])[1] - min(label, key=lambda x: x[1])[1] >= 0.5 * (max(labels[0], key=lambda x: x[1])[1] - min(labels[0], key=lambda x: x[1])[1])]
            
            if adjusted_labels:
                sub_images.append(sub_image)
                sub_labels.append(adjusted_labels)
    
    return sub_images, sub_labels


def load_save_imgs_labels(image_files,text_files,  img_out, lbl_out, YOLO=False):
    # Load images and corresponding text data
    for i in range (len(image_files)):
        # Load image
        image = cv2.imread(image_files[i])
        if image is not None:
            # Write image
            image_path = os.path.join(img_out, f'{i+1}.jpg')
            cv2.imwrite(image_path, image)
        
            # Load text data if corresponding text file exists
            # Write labels
            label_path = os.path.join(lbl_out, f'{i+1}.jpg.txt')
            output_string = ""
            for j in range(len(text_files[i])):
                data = text_files[i][j]
                # Flatten the list and format coordinates
                formatted_data = [str(coord) for point in data for coord in point]
                # Join the coordinates with commas
                output_string += ",".join(formatted_data)
                output_string += ","+str(0)
                if (j != len(text_files[i])-1):
                    output_string += '\n'
            with open(label_path, 'w') as text_output_file:
                text_output_file.write(output_string)


#Convert JSON with pascal format to the absolute format
def convert_annotations(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # If data is a dictionary, we need to extract the list of annotations
    if isinstance(data, dict):
        annotations = [data]
    else:
        annotations = data

    with open(output_file, 'w') as f:
        for i in data['shapes']:
            print(i)
            points = i["points"]
            print(points)  # Debugging: print the points
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            
            # Define the four corners of the rectangle
            coords = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ]
            
            # Flatten the list of coordinates and convert to string
            coords_flat = ','.join([f"{int(x)},{int(y)}" for x, y in coords])
            label = i.get('label', '0')  # Default to '0' if no label is provided
            
            f.write(f"{coords_flat},{label}\n")

def copy_and_rename_images(source_dir, target_dir, j):
     # Ensure target directory exists, create if not
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    
    source_image_dir = source_dir + "img/"
    source_label_dir = source_dir + "ann/"
    target_dir_train = target_dir + 'train_images/'
    target_dir_train_gts = target_dir + 'train_gts/'

    target_dir_test = target_dir + 'test_images/'
    target_dir_test_gts = target_dir + 'test_gts/'

    
    if not os.path.exists(target_dir_train):
        os.makedirs(target_dir_train)
    
    if not os.path.exists(target_dir_train_gts):
        os.makedirs(target_dir_train_gts)

    if not os.path.exists(target_dir_test):
        os.makedirs(target_dir_test)

    if not os.path.exists(target_dir_test_gts):
        os.makedirs(target_dir_test_gts)
    
    
    image_files = [f for f in os.listdir(source_image_dir)]
    image_files.sort()  # Sort files to maintain order

    k = j
    
    for i, image_file in enumerate(image_files):
        src_image_path = os.path.join(source_image_dir, image_file)
        
        if not os.path.exists(source_label_dir+ image_file + ".txt"):
            continue
        src_label_path = os.path.join(source_label_dir, image_file + ".txt")
        
        # Generate new file names
        new_image_name = f"{k+1}.jpg"
        new_label_name = f"{k+1}.jpg.txt"


        if (i < len(image_files) - 140):
            # Copy images into train
            shutil.copy(src_image_path, os.path.join(target_dir_train, new_image_name))
            shutil.copy(src_label_path, os.path.join(target_dir_train_gts, new_label_name))
            with open(os.path.join(target_dir, 'train_list.txt'), 'a') as f:
                f.write(new_image_name+"\n")
        else:
            # Copy images into test
            shutil.copy(src_image_path, os.path.join(target_dir_test, new_image_name))
            shutil.copy(src_label_path, os.path.join(target_dir_test_gts, new_label_name))
            with open(os.path.join(target_dir, 'test_list.txt'), 'a') as f:
                f.write(new_image_name+"\n")
        
        
        
        print('processed: ', image_file)    
        k += 1

    return k + len(image_files)



#creating main label list
#filename without extension
def wtite_labels_main(number, out, filename):
    filename += '.txt'
    label_path = os.path.join(out, filename)
    output_string = ""
    for j in range(1,number+1):
        output_string += str(j)+".jpg"
        if (j != number):
            output_string += '\n'
    with open(label_path, 'w') as text_output_file:
        text_output_file.write(output_string)

def get_file_names(target_path):
    file_paths = []
    file_names = []
    for file_name in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file_name)):
            file_names.append(file_name)
            file_paths.append(os.path.join(target_path, file_name))
    return file_names