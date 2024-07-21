import os
import os.path as osp
import datetime


class DBConfig(object):

    STEPS_PER_EPOCH = 750

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 80

    # Backbone network architecture
    # Supported values are: ResNet50
    BACKBONE = "ResNet50"


    # train
    EPOCHS = 100
    INITIAL_EPOCH = 0
    PRETRAINED_MODEL_PATH = './weights/best.h5'
    LOG_DIR = 'datasets/logs'
    CHECKPOINT_DIR = 'checkpoints'
    LEARNING_RATE = 1e-5


    # dataset
    IGNORE_TEXT = ["*", "###"]

    # TRAIN_DATA_PATH = 'datasets/handwritten_text_detection_splited/'
    # VAL_DATA_PATH = 'datasets/handwritten_text_detection_splited/'

    
    # TRAIN_DATA_PATH = 'datasets/TD500/'
    # VAL_DATA_PATH = 'datasets/TD500/'

    # #EVALUATION
    # TRAIN_DATA_PATH = 'evaluation dataset/after/'
    # VAL_DATA_PATH = 'evaluation dataset/after/'

    #New Document dataset
    TRAIN_DATA_PATH = 'datasets/Documents_after/'
    VAL_DATA_PATH = 'datasets/Documents_after/'
    
    
    IMAGE_SIZE = 640
    BATCH_SIZE = 2

    MIN_TEXT_SIZE = 8
    SHRINK_RATIO = 0.4

    THRESH_MIN = 0.3
    THRESH_MAX = 0.7


    def __init__(self):
        """Set values of computed attributes."""

        if not osp.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)

        self.CHECKPOINT_DIR = osp.join(self.CHECKPOINT_DIR, str(datetime.date.today()))
        if not osp.exists(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)

