from config import OCRConfig
from generate import generate
from model import OCR_Model, get_prediction_model
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


cfg = OCRConfig()

# Callbacks
def calculate_edit_distance(labels, predictions):
    sparse_labels = tf.sparse.from_dense(tf.cast(labels, tf.int64))
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    
    predictions_decoded = keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=False, beam_width=100,
    )[0][0][:, :cfg.MAX_LEN]
    
    sparse_predictions = tf.sparse.from_dense(predictions_decoded)
    
    edit_distances = tf.edit_distance(
        sparse_predictions, sparse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)

class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, pred_model, val_generator):
        super().__init__()
        self.prediction_model = pred_model
        self.generator = val_generator

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []
        #global variables of edit_distance
        total_edit_distance = 0.0
        num_samples = 0
        print("\n")
        for i in range(cfg.VALIDATION_STEPS):
            batch_images, batch_labels = next(self.generator)
            batch_predictions = self.prediction_model.predict(batch_images, verbose=0)
            
            batch_edit_distance = calculate_edit_distance(batch_labels, batch_predictions).numpy()
            total_edit_distance += batch_edit_distance
            
            num_samples += batch_images.shape[0]

            #print checkpoint
            if (i % 10):
                print(f'calculating edit distance {i}/{cfg.VALIDATION_STEPS}')

        mean_edit_distance = total_edit_distance / num_samples
        print(f'avg mean edit distance: {mean_edit_distance}')
        return mean_edit_distance

# early_stopping = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=cfg.PATIENCE,
#     restore_best_weights=True
# )

# model_checkpoint = keras.callbacks.ModelCheckpoint(
#     cfg.PRETRAINED_MODEL_PATH, 
#     monitor='val_loss', 
#     save_best_only=True
# )

# #our model
# model = OCR_Model(cfg)
# model.load_weights(cfg.PRETRAINED_MODEL_PATH)

# #Define generators
# train_generator = generate(cfg, 'train', True)
# val_generator = generate(cfg, 'val', True)
# val_generator_edit_dis = generate(cfg, 'val')

# prediction_model = get_prediction_model(model)


# #freezing layers to dense and ctc
# # Iterate over the layers of the model
# value = True
# for i in range(len(model.layers)):
#     model.layers[i].trainable = value
#     print(f"Freezing layer: {model.layers[i].name} : {value}")
#     if model.layers[i].name == 'dense2': 
#         value = False
        

# edit_distance_callback = EditDistanceCallback(prediction_model, val_generator_edit_dis)

   
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate = cfg.INIT_LEARNING_RATE,
#     decay_steps = cfg.DECAY_STEPS,
#     decay_rate = cfg.DECAY_RATE
# )


# model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

##training
# history = model.fit(
#     x = train_generator,
#     steps_per_epoch=cfg.STEPS_PER_EPOCH,
#     epochs=cfg.EPOCHS,
#     verbose=1,
#     callbacks=[early_stopping, model_checkpoint, edit_distance_callback],
#     validation_data=val_generator,
#     validation_steps=cfg.VALIDATION_STEPS
# )


##Evaluation
# test_generator = generate(cfg, 'test', True)
# results = model.evaluate(x= test_generator, steps=cfg.TEST_STEPS, verbose=1)

# # Print the results
# print(results)