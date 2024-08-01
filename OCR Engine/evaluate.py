import numpy as np
from inference import decode_result, decode_batch_predictions
import keras
from model import CTCLayer
from generate import generate
import Levenshtein

def calculate_crr(ground_truth_texts, predicted_texts):
    total_chars = sum(len(gt) for gt in ground_truth_texts)
    edit_distance_sum = sum(Levenshtein.distance(gt, pred) for gt, pred in zip(ground_truth_texts, predicted_texts))
    crr = 1 - (edit_distance_sum / total_chars)
    return crr

def calculate_wrr(ground_truth_texts, predicted_texts):
    total_words = sum(len(gt.split()) for gt in ground_truth_texts)
    correctly_identified_words = 0
    
    for gt, pred in zip(ground_truth_texts, predicted_texts):
        gt_words = gt.lower().strip().split()
        pred_words = pred.lower().strip().split()
        
        # Compare each word in ground truth with corresponding word in prediction
        for gt_word, pred_word in zip(gt_words, pred_words):
            if gt_word == pred_word:
                correctly_identified_words += 1
    
    wrr = correctly_identified_words / total_words if total_words > 0 else 0
    return wrr

# valid_aug = False
# test_generator = generate(cfg, 'test',valid_aug=valid_aug, get_only_specific_charachters=0)

# num_batches = 100
# batch_images = []
# batch_labels = []

# for _ in range(num_batches):
#     if(_%20 == 0):
#         print(_)
#     images, labels = next(test_generator)
#     batch_images.extend(images)
#     batch_labels.extend(labels)

# batch_images = np.array(batch_images)
# batch_labels = np.array(batch_labels)

# decoded_labels = decode_result(batch_labels)
# # i = 0
# # for file in file_list:
# inference_model = None
# inference_model = keras.models.load_model(osp.join(cfg.PRETRAINED_MODEL_DIR,'43.8194.h5'), custom_objects={'CTCLayer': CTCLayer})
# decoded_preds = []
# # Use this model for predictions
# preds = inference_model.predict([batch_images, batch_labels], verbose=0)
# decoded_preds = decode_batch_predictions(preds)

# crr = calculate_crr(decoded_labels, decoded_preds)
# wrr = calculate_wrr(decoded_labels, decoded_preds)

# print(f'scored crr:{crr}, wrr:{wrr} on {cfg.MAX_NUMBER_OF_WORDS} words, with {valid_aug} validation\n')