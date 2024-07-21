import math
import cv2
import os.path as osp
import glob
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import config

# from model import dbnet


def resize_image(image, image_short_side=736):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def box_score_fast(bitmap, _box):
    # 计算 box 包围的区域的平均得分
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int_), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int_), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int_), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int_), 0, h - 1)

    # #testing
    # print("height: ", h)
    # print("width: ", w)
    # print("box[:, 0]: ", box[:, 0] )
    # print("box[:, 1]: ", box[:, 1] )
    # print("np.floor(box[:, 0].min()).astype(np.int_): ", np.floor(box[:, 0].min()).astype(np.int_))
    # print("w-1 : ", w-1)
    # print("np.clip(np.floor(box[:, 0].min()).astype(np.int_), 0, w - 1): ",np.clip(np.floor(box[:, 0].min()).astype(np.int_), 0, w - 1))


    # print("x_min: ", xmin)
    # print("x_max: ", xmax)
    # print("y_min: ", ymin)
    # print("y_max: ", ymax)

    #creating a mask of zeros with the same shape as the contour
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

    # #testing
    # print("mask: ", mask)
    # print("mask shape: ", mask.shape)
    
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin

    # #testing
    # print("box[:, 0] - xmin: ", box[:, 0])
    # print("box[:, 1] - ymin: ", box[:, 1])

    # print("box.shape: ", box.shape)
    # print("box.reshape(1,-1,2): ", box.reshape(1, -1, 2).shape)
    
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

    
    # print("bitmap[ymin:ymax + 1, xmin:xmax + 1].shape: ", bitmap[ymin:ymax + 1, xmin:xmax + 1].shape)
    # print("prediction .shape: ", bitmap.shape)

    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


#testing
# Function to check if two contours overlap
def doContoursOverlap(cnt1, cnt2):
    x, y, w, h = cv2.boundingRect(cnt1)
    rect1 = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    x, y, w, h = cv2.boundingRect(cnt2)
    rect2 = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    overlap_area = cv2.contourArea(cv2.convexHull(np.vstack((cnt1, cnt2))))
    area1 = cv2.contourArea(cnt1)
    area2 = cv2.contourArea(cnt2)

    return overlap_area > min(area1, area2) * 0.5  # Adjust threshold as needed


# Function to combine overlapping contours
def combineContours(contours):
    merged_contours = list(contours)
    i = 0
    while i < len(merged_contours):
        j = i + 1
        while j < len(merged_contours):
            if doContoursOverlap(merged_contours[i], merged_contours[j]):
                # Merge contours
                merged_contours[i] = np.vstack((merged_contours[i], merged_contours[j]))
                merged_contours.pop(j)
            else:
                j += 1
        i += 1
    return merged_contours


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=10000, box_thresh=0.7, image_fname=""):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    
    # cv2.imwrite('test/bitmap threshold experiment/' + image_fname, bitmap*255)

    #testing different modes
    #Cannot oberve any difference between the modes, 
    #he must haven't used the hirarcal structure of the contours
    # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #testing different methods
    # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    # contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    # #Decided to use L1 method 
    # gray = (bitmap * 255).astype(np.uint8)
    
    # Find contours in the bitmap image
    contours, _ = cv2.findContours((bitmap).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # print(len(contours))
    
    # contours = combineContours(contours)

    # contours2 = []
    
    # print(len(contours))
    
    for contour in contours:
        
        
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        
        
        # #to remove any countours that has less than 4 dims
        # if points.shape[0] < 4:
        #     continue

        # #getting the mean prediction values of the ROI we put in
        score = box_score_fast(pred, points.reshape(-1, 2))
        
        # #testing
        # print("contour ", contour.shape)
        # #testing
        # print("arcLength= ", cv2.arcLength(contour, True))
        # print("epsilon: ", epsilon)
        # print("approx: ", approx)
        # print("before approx.reshape: ", approx.shape)
        # print("approx.reshape: ", approx.reshape((-1, 2)).shape)
        # #testing
        # print("score: ", score)
        # break;
        
        if box_thresh > score:
            continue

        # print("points", points)
        
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=1.5)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
        # contours2.append(contour)

    # Create a blank image to draw contours on
    # contour_image = np.zeros((736, 1120, 3), dtype=np.uint8)  # Create a blank image with 3 channels (RGB)
    # Draw contours on the blank image
    # cv2.drawContours(contour_image, co, -1, (0, 255, 0), 2)
    # cv2.imwrite('test/random/X-' + image_fname, contour_image)

    return boxes, scores


def split_image_with_overlap(image, overlap_factor = 0.6):
    # Get dimensions of the image
    height, width, _ = image.shape
    
    
    # Calculate the dimensions for each cropped piece
    crop_height = height // 2
    crop_width = width // 2
    
    # Calculate overlap sizes
    overlap_height = int(crop_height * overlap_factor)
    overlap_width = int(crop_width * overlap_factor)
    
    cropped_pieces = []
    
    # Loop through rows and columns to crop the image
    for y in range(0, height - crop_height + 1, crop_height - overlap_height):
        for x in range(0, width - crop_width + 1, crop_width - overlap_width):
            # Define the cropping region
            start_x = x
            start_y = y
            end_x = min(x + crop_width, width)
            end_y = min(y + crop_height, height)
            
            # Crop the image
            cropped_piece = image[start_y:end_y, start_x:end_x]
            
            # Append the cropped piece to the list
            cropped_pieces.append(cropped_piece)
    print(len(cropped_pieces))
    return cropped_pieces


def process(image, p_model, image_path, output_path, output_bitmap, out_boxes = False, return_boxes =False):
    
    mean = np.array([103.939, 116.779, 123.68])
    
    src_image = image.copy()
    h, w = image.shape[:2]
    image = resize_image(image)
    image = image.astype(np.float32)
    image -= mean
    image_input = np.expand_dims(image, axis=0)
    p = p_model.predict(image_input, verbose=0)[0]
    
    # print(len(p[0]))
    # print(w, h)

    threshold = 0.05
    image_fname = osp.split(image_path)[-1]


    
    boxes, scores = polygons_from_bitmap(p, p < threshold, w, h,box_thresh=0.3, image_fname=image_fname[0]+"-0.04-"+image_fname)


    if (out_boxes):
        split_image_by_boxes(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes, "./test/final/")
    # print(len(boxes))
    # print(scores)

    if (return_boxes):
        return boxes

    
    for box in boxes:
        cv2.drawContours(src_image, [np.array(box)], -1, (0, 255, 0), 2)
        
    cv2.imwrite(output_path + image_fname, src_image)

    if (output_bitmap == True):
        cv2.imwrite(output_path+'bitmap-' + image_fname, (p<threshold)*255)

    return src_image
        
        

def callProcess(img, p_model, mode='default', path = 'test/random/', bitmap = False):
    
    process(img, "input/"+"-"+ mode +".png", path, bitmap)
    cropped_pieces = split_image_with_overlap(img)
    for i in range(len(cropped_pieces)):
        process(cropped_pieces[i], p_model, "input/"+str(i)+"-"+ mode +".png", path, bitmap)


'''
if __name__ == '__main__':
    cfg = DBConfig()
    p_model = DBNet(cfg, model='inference')
    model.load_weights(cfg.PRETRAINED_MODEL_PATH, by_name=True, skip_mismatch=True)

    image = cv2.imread("input/59.JPG")
    mode = '10-'
    path = 'test/'
    callProcess(image, p_model, mode, path, True)

'''
