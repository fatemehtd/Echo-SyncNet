
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ******************************************************************************
# Experiment params
# ******************************************************************************

# Directory for the experiment logs.
CONFIG.LOGDIR = '../../../tmp/alignment_logs/'
# Dataset for training alignment.
# Check dataset_splits.py for full list.
CONFIG.DATASETS = [
    'shovel'
]

# Path to tfrecords.
CONFIG.PATH_TO_TFRECORDS = '/bigdata/DataSet/Intelliperception_data/%s_tfrecords/'
# Algorithm used for training: alignment, sal, alignment_sal_tcn,
# classification, tcn . (alignment is called tcc in paper)
CONFIG.TRAINING_ALGO = 'alignment_sal_tcn'
# Size of images/frames.
CONFIG.IMAGE_SIZE = 64  # For ResNet50

# ******************************************************************************
# Training params
# ******************************************************************************

# Number of training steps.
CONFIG.TRAIN = edict()
CONFIG.TRAIN.MAX_ITERS = 300000
CONFIG.TRAIN.MAX_EPOCHS = 100
# Number of samples in each batch.
CONFIG.TRAIN.BATCH_SIZE = 2
# Number of frames to use while training.
CONFIG.TRAIN.NUM_FRAMES = 100
CONFIG.TRAIN.VISUALIZE_INTERVAL = 500

# ******************************************************************************
# Eval params
# ******************************************************************************
CONFIG.EVAL = edict()
# Number of samples in each batch.S
CONFIG.EVAL.BATCH_SIZE = 2
# Number of frames to use while evaluating. Only used to see loss in eval mode.
CONFIG.EVAL.NUM_FRAMES = 100

CONFIG.EVAL.VAL_ITERS = 20
# A task evaluates the embeddings or the trained model.
# Currently available tasks are: 'algo_loss', 'classification',
# 'kendalls_tau', 'event_completion' (called progression in paper),
# 'few_shot_classification'
# Provide a list of tasks using which the embeddings will be evaluated.
CONFIG.EVAL.TASKS = [
    'algo_loss',
    'classification',
    'kendalls_tau'
    # 'event_completion',
    # 'few_shot_classification'
]
CONFIG.EVAL.VISUALIZE = False
CONFIG.EVAL.FRAMES_PER_BATCH = 25
CONFIG.EVAL.KENDALLS_TAU_STRIDE = 3  # 2 for Pouring, 5 for PennAction
CONFIG.EVAL.KENDALLS_TAU_DISTANCE = 'sqeuclidean'  # cosine, sqeuclidean
CONFIG.EVAL.CLASSIFICATION_FRACTIONS = [0.1, 0.5, 1.0]
CONFIG.EVAL.LINEAR_MODEL_TYPE = 'svc' # 'svr' for regression

# If number this is considered to
CONFIG.EVAL.FEW_SHOT_NUM_LABELED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CONFIG.EVAL.FEW_SHOT_NUM_EPISODES = 50


# ******************************************************************************
# Model params
# ******************************************************************************
# Currently InceptionV3 implies load ImageNet pretrained weights.
CONFIG.MODEL = edict()

CONFIG.MODEL.EMBEDDER_TYPE = 'conv'

CONFIG.MODEL.BASE_MODEL = edict()
# Resnet50, VGGM
CONFIG.MODEL.BASE_MODEL.NETWORK = 'Resnet50_pretrained'
# conv4_block3_out, conv4 (respective layers in networks)
CONFIG.MODEL.BASE_MODEL.LAYER = 'conv4_block3_out'

# Select which layers to train.
# train_base defines how we want proceed with fine-tuning the base model.
# 'frozen' : Weights are fixed and batch_norm stats are also fixed.
# 'train_all': Everything is trained and batch norm stats are updated.
# 'only_bn': Only tune batch_norm variables and update batch norm stats.
CONFIG.MODEL.TRAIN_BASE = 'train_all'
CONFIG.MODEL.TRAIN_EMBEDDING = True

# pylint: disable=line-too-long
CONFIG.MODEL.RESNET_PRETRAINED_WEIGHTS = "/mnt/nas/workspace/fatemeht/projects/pre-trained_ResNet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
# pylint: enable=line-too-long

# VGG_M-esque model
CONFIG.MODEL.VGGM = edict()
CONFIG.MODEL.VGGM.USE_BN = True

CONFIG.MODEL.CONV_EMBEDDER_MODEL = edict()
# List of conv layers defined as (channels, kernel_size, activate).
CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS = [
    (256, 3, True),
    (256, 3, True),
]
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD = 'max_pool'
# List of fc layers defined as (channels, activate).
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS = [
    (256, True),
    (256, True),
]
CONFIG.MODEL.CONV_EMBEDDER_MODEL.CAPACITY_SCALAR = 2
CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE = 128
CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE = False
CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_RATE = 0.0
CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_SPATIAL = False
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_DROPOUT_RATE = 0.1
CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN = True

# Conv followed by GRU Embedder
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL = edict()
# List of conv layers defined as (channels, kernel_size, activate).
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.CONV_LAYERS = [(512, 3, True),
                                                   (512, 3, True)]
# List of fc layers defined as (channels, activate).
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.GRU_LAYERS = [
    128,
]
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.DROPOUT_RATE = 0.0
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.USE_BN = True

CONFIG.MODEL.L2_REG_WEIGHT = 0.00001

# ******************************************************************************
# Alignment params
# ******************************************************************************
CONFIG.ALIGNMENT = edict()
CONFIG.ALIGNMENT.CYCLE_LENGTH = 2
CONFIG.ALIGNMENT.LABEL_SMOOTHING = 0.1
CONFIG.ALIGNMENT.SOFTMAX_TEMPERATURE = 1
# CONFIG.ALIGNMENT.LOSS_TYPE = 'regression_mse_var'
CONFIG.ALIGNMENT.LOSS_TYPE = 'classification'
CONFIG.ALIGNMENT.NORMALIZE_INDICES = True
CONFIG.ALIGNMENT.VARIANCE_LAMBDA = 0.001
CONFIG.ALIGNMENT.FRACTION = 1.0
CONFIG.ALIGNMENT.HUBER_DELTA = 0.1
CONFIG.ALIGNMENT.SIMILARITY_TYPE = 'l2'  # l2, cosine
# Stochastic matching is not optimized for TPUs.
# Initial experiments were done with stochastic version, which can potentially
# handle longer sequences.
CONFIG.ALIGNMENT.STOCHASTIC_MATCHING = False

# ******************************************************************************
# Shuffle and Learn params
# ******************************************************************************
CONFIG.SAL = edict()
CONFIG.SAL.DROPOUT_RATE = 0.0
# List of fc layers defined as (channels, activate).
CONFIG.SAL.FC_LAYERS = [(128, True), (64, True), (2, False)]
CONFIG.SAL.SHUFFLE_FRACTION = 0.75
# Number of triplets to sample from each video in batch.
CONFIG.SAL.NUM_SAMPLES = 8
CONFIG.SAL.LABEL_SMOOTHING = 0.0

# ******************************************************************************
# Alignment and Shuffle and Learn and TCN params
# ******************************************************************************
CONFIG.ALIGNMENT_SAL_TCN = edict()
# The weight for the tcn loss is (1 - alignment_loss_weight - sal_loss_weight)
CONFIG.ALIGNMENT_SAL_TCN.ALIGNMENT_LOSS_WEIGHT = 0.33
CONFIG.ALIGNMENT_SAL_TCN.SAL_LOSS_WEIGHT = 0.33

# ******************************************************************************
# Classification/Supervised Learning of Per-frame Classes params
# Classification/Supervised Learning of Per-frame Classes params
# ******************************************************************************
CONFIG.CLASSIFICATION = edict()
CONFIG.CLASSIFICATION.LABEL_SMOOTHING = 0.0
CONFIG.CLASSIFICATION.DROPOUT_RATE = 0.0

# ******************************************************************************
# Time Contrastive Network params
# ******************************************************************************
CONFIG.TCN = edict()
CONFIG.TCN.POSITIVE_WINDOW = 5
CONFIG.TCN.REG_LAMBDA = 0.002

# ******************************************************************************
# Optimizer params
# ******************************************************************************
CONFIG.OPTIMIZER = edict()
# Supported optimizers are: AdamOptimizer, MomentumOptimizer
CONFIG.OPTIMIZER.TYPE = 'AdamOptimizer'

CONFIG.OPTIMIZER.LR = edict()
# Initial learning rate for optimizer.
CONFIG.OPTIMIZER.LR.INITIAL_LR = 0.0001
# Learning rate decay strategy.
# Currently Supported strategies: fixed, exp_decay, manual
CONFIG.OPTIMIZER.LR.DECAY_TYPE = 'fixed'
CONFIG.OPTIMIZER.LR.EXP_DECAY_RATE = 0.97
CONFIG.OPTIMIZER.LR.EXP_DECAY_STEPS = 1000
CONFIG.OPTIMIZER.LR.MANUAL_LR_STEP_BOUNDARIES = [5000, 10000]
CONFIG.OPTIMIZER.LR.MANUAL_LR_DECAY_RATE = 0.1
CONFIG.OPTIMIZER.LR.NUM_WARMUP_STEPS = 0

# ******************************************************************************
# Data params
# ******************************************************************************
CONFIG.DATA = edict()
CONFIG.DATA.SHUFFLE_QUEUE_SIZE = 100
CONFIG.DATA.NUM_PREFETCH_BATCHES = 1
CONFIG.DATA.RANDOM_OFFSET = 1
CONFIG.DATA.STRIDE = 15 # used in 'stride' strategy
CONFIG.DATA.SAMPLING_STRATEGY = 'offset_uniform'  # offset_uniform, stride
CONFIG.DATA.NUM_STEPS = 2  # number of frames that will be embedded jointly,
CONFIG.DATA.FRAME_STRIDE = 15  # stride between context frames
# Set this to False if your TFRecords don't have per-frame labels.
CONFIG.DATA.FRAME_LABELS = True
CONFIG.DATA.PER_DATASET_FRACTION = 1.0  # Use 0 to use only one sample.
CONFIG.DATA.PER_CLASS = False
# stride of frames while embedding a video during evaluation.
CONFIG.DATA.SAMPLE_ALL_STRIDE = 5

# ******************************************************************************
# Augmentation params
# ******************************************************************************
CONFIG.AUGMENTATION = edict()
CONFIG.AUGMENTATION.RANDOM_FLIP = False
CONFIG.AUGMENTATION.RANDOM_CROP = False
CONFIG.AUGMENTATION.BRIGHTNESS = False
CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA = 32.0 / 255
CONFIG.AUGMENTATION.CONTRAST = False
CONFIG.AUGMENTATION.CONTRAST_LOWER = 0.5
CONFIG.AUGMENTATION.CONTRAST_UPPER = 1.5
CONFIG.AUGMENTATION.HUE = False
CONFIG.AUGMENTATION.HUE_MAX_DELTA = 0.2
CONFIG.AUGMENTATION.SATURATION = False
CONFIG.AUGMENTATION.SATURATION_LOWER = 0.5
CONFIG.AUGMENTATION.SATURATION_UPPER = 1.5

# ******************************************************************************
# Logging params
# ******************************************************************************
CONFIG.LOGGING = edict()
# Number of steps between summary logging.
CONFIG.LOGGING.REPORT_INTERVAL = 100

# ******************************************************************************
# Checkpointing params
# ******************************************************************************
CONFIG.CHECKPOINT = edict()
# Number of steps between consecutive checkpoints.
CONFIG.CHECKPOINT.SAVE_INTERVAL = 1000
# IF TRAINING IS OPTICALFLOW
CONFIG.OPTICALFLOW = False
CONFIG.SEED = 4831
