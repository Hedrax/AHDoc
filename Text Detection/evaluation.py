import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate the IoU of two quadrilateral bounding boxes.
    box1, box2: list of tuples, e.g., [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    def polygon_area(points):
        x, y = zip(*points)
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    # Convert boxes to numpy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Calculate area of each polygon
    area1 = polygon_area(box1)
    area2 = polygon_area(box2)
    
    # Find the intersection polygon
    from shapely.geometry import Polygon
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    if not poly1.intersects(poly2):
        return 0.0
    
    inter_area = poly1.intersection(poly2).area
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area

def evaluate_boxes(pred_boxes, true_boxes, iou_threshold=0.5):
    TP = 0
    FP = 0
    FN = 0
    iou_scores = []
    
    matched_true_boxes = set()
    
    for pred in pred_boxes:
        best_iou = 0
        best_true_idx = -1
        for idx, true in enumerate(true_boxes):
            iou = calculate_iou(pred, true)
            if iou > best_iou:
                best_iou = iou
                best_true_idx = idx
        
        if best_iou >= iou_threshold:
            if best_true_idx not in matched_true_boxes:
                TP += 1
                matched_true_boxes.add(best_true_idx)
                iou_scores.append(best_iou)
            else:
                FP += 1
        else:
            FP += 1
    
    FN = len(true_boxes) - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    
    return precision, recall, f1, avg_iou
    

