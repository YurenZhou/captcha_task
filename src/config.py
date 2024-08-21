
class CONFIG:

    IMAGE_PATH = "sampleCaptchas/input"
    LABEL_PATH = "sampleCaptchas/output"
    TRAINED_MODEL_PATH = "models/model.pth"
    LABEL_ENCODER_PATH = "models/label_encoder.pkl"
    BEST_HYPER_PARAM_PATH = "models/best_hyper_parameters.pkl"

    BATCH_SIZE = 32
    NUM_EPOCH = 10
    HYPER_TUNING_TRIALS = 5

    LEARNING_RATE_LOWER = 0.01
    LEARNING_RATE_UPPER = 0.05
    NUM_FILTERS_LOWER = 4
    NUM_FILTERS_UPPER = 8
    DROPOUT_RATE_LOWER = 0.2
    DROPOUT_RATE_UPPER = 0.4

