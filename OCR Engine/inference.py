import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from config import OCRConfig
import arabic_reshaper
from bidi.algorithm import get_display
from generate import generate
from model import OCR_Model



cfg = OCRConfig()

def decode_result(results):
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, 99)))[:, 0]
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))[:, 0]
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False, beam_width=50)[0][0][
        :, :cfg.MAX_LEN
    ]
    
    output_text = decode_result(results)
    return output_text

# test_generator = generate(cfg, 'test')
# # Define the inference model
# inference_model = OCR_Model(cfg, 'inference')
# # Assuming you've loaded weights and they are compatible up to the inference output layer
# inference_model.load_weights(cfg.PRETRAINED_MODEL_PATH, by_name=True)  # by_name=True helps align the layer names and weights

# # Use this model for predictions
# batch_images, batch_labels = next(test_generator)

num_to_char = StringLookup(vocabulary=cfg.characters, mask_token=None, invert=True)


# preds = inference_model.predict(batch_images, verbose=0)
# decoded_labels, decoded_preds = decode_result(batch_labels), decode_batch_predictions(preds)
# print(decoded_preds[0])

# fig, axes = plt.subplots(4, 4, figsize=(23, 10))

# for i, ax in enumerate(axes.flatten()):
#     label = get_display( arabic_reshaper.reshape(decoded_labels[i]))
#     pred = get_display( arabic_reshaper.reshape(decoded_preds[i]))
#     ax.imshow(batch_images[i])
#     ax.axis('off')
#     ax.set_title(f"{label}\n{pred}", fontdict=None)


# # Adjust spacing
# plt.subplots_adjust(wspace=0.1, hspace=0.5)  # Adjust these values as needed

# plt.show()