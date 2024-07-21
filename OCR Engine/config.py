class OCRConfig(object):
    #Configuration file
    
    IMAGE_WIDTH = 508
    IMAGE_HEIGHT = 64
    
    
    #paths
    TRAIN_DATA_PATH = './dataset/train/'
    VALIDATION_DATA_PATH = './dataset/val/'
    TEST_DATA_PATH = './dataset/test/'
    
    PRETRAINED_MODEL_PATH = './weights/best.h5'
    
    
    
    #model config
    INIT_LEARNING_RATE = 0.0005
    DECAY_STEPS = 10000
    DECAY_RATE = 0.9
    PATIENCE_VALUE = 6
    
    
    #compile
    RESTORE_BEST_WEIGHTS = True
    SAVE_BEST_MODEL = True
    
    
    #train
    BATCH_SIZE = 32
    EPOCHS = 100
    STEPS_PER_EPOCH = 1844
    VALIDATION_STEPS = 32
    PATIENCE = 10
    
    #generate
    MAX_LEN = 106
    MAX_NUMBER_OF_WORDS = 10
    PADDING_TOKEN = 99  
    
    CLEAN_DATA_PERCENTAGE = 70

    #test
    TEST_STEPS = 32
    MAX_NOISE_APPLIED = 60
    DILUTE_KERNAL_FACTOR = 2


    #augmantation
    MAX_ROTATION_ANGLE = 4
    
    NUM_OF_DROPS = 400
    DROPS_RADIUS = 1
    DROPS_ALPHA = 0.5

    POLYGON_ALPHA = 0.5
    POLYGON_COLOR = (0, 0, 0)
    
    characters = [' ', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ'
                  , 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع'
                  , 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', 'ـ', '[blank]']

