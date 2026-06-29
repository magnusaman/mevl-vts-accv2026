from detectron2.config import CfgNode as CN

def add_gom_config(cfg):
    _C = cfg

    # association head
    _C.MODEL.ASSO_ON = False
    _C.MODEL.ASSO_HEAD = CN() # tracking transformer architecture parameters
    _C.MODEL.ASSO_HEAD.FC_DIM = 1024 
    _C.MODEL.ASSO_HEAD.NUM_FC = 2
    _C.MODEL.ASSO_HEAD.NUM_ENCODER_LAYERS = 1
    _C.MODEL.ASSO_HEAD.NUM_DECODER_LAYERS = 1
    _C.MODEL.ASSO_HEAD.NUM_WEIGHT_LAYERS = 2
    _C.MODEL.ASSO_HEAD.NUM_HEADS = 8
    _C.MODEL.ASSO_HEAD.DROPOUT = 0.1
    _C.MODEL.ASSO_HEAD.NORM = False
    _C.MODEL.ASSO_HEAD.ASSO_THRESH = 0.1
    _C.MODEL.ASSO_HEAD.ASSO_WEIGHT = 1.0
    _C.MODEL.ASSO_HEAD.NEG_UNMATCHED = False
    _C.MODEL.ASSO_HEAD.NO_DECODER_SELF_ATT = True
    _C.MODEL.ASSO_HEAD.NO_ENCODER_SELF_ATT = False
    _C.MODEL.ASSO_HEAD.WITH_TEMP_EMB = False
    _C.MODEL.ASSO_HEAD.NO_POS_EMB = False
    _C.MODEL.ASSO_HEAD.ASSO_THRESH_TEST = -1.0
    _C.MODEL.ASSO_HEAD.CTRS_WEIGHT = 1.0
    _C.MODEL.ASSO_HEAD.ASSO_WEIGHT_LOCAL = 1.0
    _C.MODEL.ROI_HEADS.WITH_RESR = True

    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.SIZE = 'B' # 'T', 'S', 'B'
    _C.MODEL.SWIN.USE_CHECKPOINT = False
    _C.MODEL.SWIN.OUT_FEATURES = (1, 2, 3) # (0, 1, 2, 3)

    _C.MODEL.FREEZE_TYPE = ''
    
    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1
    _C.SOLVER.USE_CUSTOM_SOLVER = False
    _C.SOLVER.OPTIMIZER = 'SGD'
    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0
    _C.SOLVER.CUSTOM_MULTIPLIER = 1.0
    _C.SOLVER.CUSTOM_MULTIPLIER_NAME = []

    _C.DATALOADER.SOURCE_AWARE = False
    _C.DATALOADER.DATASET_RATIO = [1, 1]

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TRAIN_H = -1
    _C.INPUT.TRAIN_W = -1
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.TEST_H = -1
    _C.INPUT.TEST_W = -1
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 
    _C.INPUT.NOT_CLAMP_BOX = False

    _C.INPUT.VIDEO = CN()
    _C.INPUT.VIDEO.TRAIN_LEN = 8 # number of frames in training
    _C.INPUT.VIDEO.TEST_LEN = 16 # number of frames for tracking in testing
    _C.INPUT.VIDEO.SAMPLE_RANGE = 2.0 # sampling frames with a random stride 
    _C.INPUT.VIDEO.DYNAMIC_SCALE = True # Increase video length for smaller resolution
    _C.INPUT.VIDEO.GEN_IMAGE_MOTION = True # Interpolate between two augmentations
    
    _C.VIDEO_INPUT = False
    _C.VIDEO_TEST = CN()
    _C.VIDEO_TEST.OVERLAP_THRESH = 0.1 # threshold for continuing a tracking or starting a new track
    _C.VIDEO_TEST.NOT_MULT_THRESH = False # threshold for continuing a tracking or starting a new track
    _C.VIDEO_TEST.MIN_TRACK_LEN = 5 # post processing to filter out short tracks
    _C.VIDEO_TEST.MAX_CENTER_DIST = -1. # threshold for continuing a tracking or starting a new track
    _C.VIDEO_TEST.DECAY_TIME = -1. # reweighting hyper-parameters for association
    _C.VIDEO_TEST.WITH_IOU = False # combining with location in our tracker
    _C.VIDEO_TEST.LOCAL_TRACK = False # Run our baseline tracker
    _C.VIDEO_TEST.LOCAL_IOU_ONLY = False # IOU-only baseline
    _C.VIDEO_TEST.LOCAL_NO_IOU = False # ReID-only baseline
    _C.VIDEO_TEST.NMS_THRESH = 0.5

    _C.VIS_THRESH = 0.3
    _C.NOT_EVAL = False
    _C.FIND_UNUSED_PARAM = True

    # ---------------- DenseTrack v7 ----------------
    _C.MODEL.DENSETRACK = CN()
    _C.MODEL.DENSETRACK.ENABLED = False  # master switch

    # Component 1: SAM-augmented proposal recall
    _C.MODEL.DENSETRACK.COMP1_ENABLED = False
    _C.MODEL.DENSETRACK.COMP1_CACHE_ROOT = ""  # offline SAM mask cache
    # CLIP features from SAM cache are open_clip projected (768-d for ViT-L/14-336),
    # NOT raw 1024-d hidden state. Confirmed via Stage 6 smoke.
    _C.MODEL.DENSETRACK.COMP1_CLIP_DIM = 768
    _C.MODEL.DENSETRACK.COMP1_HIDDEN_DIM = 256
    _C.MODEL.DENSETRACK.COMP1_MAX_PER_FRAME = 50

    # Component 2: multi-encoder consensus matcher.
    # SAM-H dropped from the encoder set — OOMs on L4 at 1024px input. The
    # consensus matcher works fine with 3 orthogonal foundation priors
    # (CLIP=web image-text, DINOv2=self-sup geometric, ConvNeXt=ImageNet sup).
    _C.MODEL.DENSETRACK.COMP2_ENABLED = False
    _C.MODEL.DENSETRACK.COMP2_CACHE_ROOT = ""  # offline encoder feature cache
    _C.MODEL.DENSETRACK.COMP2_ENCODERS = ["clip-l-336", "dinov2-l", "convnext-l"]
    _C.MODEL.DENSETRACK.COMP2_ENCODER_DIMS = [1024, 1024, 1536]
    _C.MODEL.DENSETRACK.COMP2_PROJ_DIM = 256

    # Component 3: VLM content matching
    _C.MODEL.DENSETRACK.COMP3_ENABLED = False
    _C.MODEL.DENSETRACK.COMP3_TEXT_CACHE_ROOT = ""  # offline VLM transcription cache
    _C.MODEL.DENSETRACK.COMP3_USE_MLP_FUSION = True

    # Training: warmup iters during which v7 gates stay frozen at init (=0)
    _C.MODEL.DENSETRACK.GATE_WARMUP_ITERS = 1000