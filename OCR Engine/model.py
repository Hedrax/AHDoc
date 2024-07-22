import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as KL
from tensorflow.keras.metrics import Metric
from config import OCRConfig
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import numpy as np

cfg = OCRConfig()

# Model definition
class SequenceAccuracy(Metric):
    def __init__(self, name='sequence_accuracy', **kwargs):
        super(SequenceAccuracy, self).__init__(name=name, **kwargs)
        self.correct_predictions = self.add_weight(name='cp', initializer='zeros')
        self.total_predictions = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_length = tf.math.reduce_sum(tf.cast(tf.not_equal(y_pred, 0), tf.int64), axis=1)
        decoded_preds = keras.backend.ctc_decode(y_pred, input_length=input_length, greedy=True)[0][0]
        y_true = tf.sparse.from_dense(y_true)
        decoded_preds = tf.sparse.from_dense(decoded_preds)
        edit_distance = tf.edit_distance(decoded_preds, y_true, normalize=False)
        perfect_matches = tf.cast(tf.equal(edit_distance, 0), tf.float32)
        self.correct_predictions.assign_add(tf.reduce_sum(perfect_matches))
        self.total_predictions.assign_add(tf.size(perfect_matches))

    def result(self):
        return self.correct_predictions / self.total_predictions

    def reset_states(self):
        self.correct_predictions.assign(0)
        self.total_predictions.assign(0)


class CTCLayer(KL.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def get_prediction_model(model):
    prediction_model = keras.models.Model(
        model.get_layer(name="image_input").input, model.get_layer(name="dense3").output)
    return prediction_model



def OCR_Model(cfg, type='train'):
    
    char_to_num = StringLookup(vocabulary=cfg.characters, mask_token=None)

    
    input_img = keras.Input(shape=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3), name="image_input")
    labels = KL.Input(name="label", shape=(None,))
    
    x = KL.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1",)(input_img)
    x = KL.MaxPooling2D((2, 2), name="pool1")(x)
    x = KL.BatchNormalization()(x)
    
    new_shape = ((cfg.IMAGE_WIDTH // 2), (cfg.IMAGE_HEIGHT // 2) * 32)
    x = KL.Reshape(target_shape=new_shape, name="reshape")(x)
    
    x = KL.Dense(16, activation="relu", name="dense2")(x)
    x = KL.BatchNormalization()(x)
    
    x = KL.Bidirectional(keras.layers.LSTM(256, return_sequences=True, dropout=0.35))(x)
    
    final_dense = KL.Dense(len(cfg.characters) + 3, activation="softmax", name="dense3")(x)
    
    output = CTCLayer(name="ctc_loss")(labels, final_dense)
    if (type == 'inference'):
        model = keras.models.Model(
            inputs=input_img, outputs=final_dense, name="OCR-ENGINE"
        )
    else:
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="OCR-ENGINE"
        )
        
    
    return model
