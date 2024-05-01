import argparse
import torch


class Args(argparse.Namespace):
    batch_size = 64
    epochs = 40
    update_freq = 1
    save_ckpt_freq = 10

    # Model parameters
    model = "vit_small_patch16_224"
    tubelet_size = 2
    input_size = 224
    fc_drop_rate = 0.0
    drop = 0.0
    attn_drop_rate = 0.0
    drop_path = 0.1
    disable_eval_during_finetuning = False
    model_ema = False
    model_ema_decay = 0.9999
    model_ema_force_cpu = False

    # Optimizer parameters
    opt = "adamw"
    opt_eps = 1e-8
    opt_betas = [0.9, 0.999]
    clip_grad = None
    momentum = 0.9
    weight_decay = 0.05
    weight_decay_end = None
    lr = 1e-3
    layer_decay = 0.7
    warmup_lr = 1e-6
    min_lr = 1e-6
    warmup_epochs = 5
    warmup_steps = -1

    # Augmentation parameters
    color_jitter = 0.4
    num_sample = 2
    aa = "rand-m7-n4-mstd0.5-inc1"
    smoothing = 0.1
    train_interpolation = "bicubic"

    # Evaluation parameters
    crop_pct = None
    short_side_size = 224
    test_num_segment = 1
    test_num_crop = 3

    # Random Erase params
    reprob = 0.25
    remode = "pixel"
    recount = 1
    resplit = False

    # Mixup params
    mixup = 0.8
    cutmix = 1
    cutmix_minmax = None
    mixup_prob = 1
    mixup_switch_prob = 0.5
    mixup_mode = "batch"

    # Finetuning params
    finetune = "checkpoints/VideoMAE_ssv2.pth"
    model_key = "model|module"
    model_prefix = ""
    init_scale = 0.001
    use_checkpoint = False
    use_mean_pooling = True
    # use_cls =

    # Dataset parameters
    data_path = "ssv2"
    anno_path = "ssv2/ssv2.csv"
    ssv2_labels = "ssv2/labels.json"
    eval_data_path = None
    nb_classes = 174
    imagenet_default_mean_and_std = True
    num_segments = 1
    num_frames = 16
    sampling_rate = 4
    data_set = "SSV2"
    output_dir = "runs/"
    log_dir = "runs"
    seed = 0
    resume = ""
    auto_resume = True

    save_ckpt = True
    dist_eval = False
    num_workers = 10
    pin_mem = True

    # distributed training parameters
    world_size = 1
    local_rank = -1
    dist_url = "env://"
    enable_deepspeed = True

    # EgoHOS
    mode = "twohands_obj2"  # "cb","twohands_obj2"

    config_file_2h = "externals/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py"
    checkpoint_file_2h = "externals/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth"

    config_file_cb = "externals/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py"
    checkpoint_file_cb = "externals/EgoHOS/mmsegmentation/work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth"

    config_file_obj1 = "externals/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py"
    checkpoint_file_obj1 = "externals/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth"

    config_file_obj2 = "externals/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj2_ccda/twohands_cb_to_obj2_ccda.py"
    checkpoint_file_obj2 = "externals/EgoHOS/mmsegmentation/work_dirs/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth"

    pred_seg_dir = ["images/2h", "images/cb", "images/obj2"]

    img_dir = "images/raw"
    vis_dir = "images/HOS_results"
    cb_dir = "images/cb"
    twohands_dir = "images/2h"
    obj2_dir = "images/obj2"

    tests_dir = "images/tests"
    cb_view = "images/cb_view"

    # Interacting Object bbs from annotations
    annotation_dir = "ssv2/annotations"

    # Image dataset dirs
    obj_crop_dir = "ssv2/object_crops"
    VAE_features_dir = "ssv2/VAE_features"

    # AcE arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLIP_model = "ViT-B/32"
    split_ratios = [0.6, 0.2, 0.2]
    AcE_batch_size = 360
    AcE_feature_size = 384
    AcE_epochs = 500
    AcE_criterion = "SmoothL1Loss"  # "MSE"
    AcE_lr = 0.0001
    AcE_checkpoint = None  #  "checkpoints/AcE_head_99.pth"
    teacher_head_checkpoint = "checkpoints/teacher_head_4.pth"
    teacher_lr = 10e-4
    teacher_epochs = 5

    labels_to_keep = [
        2,  # "Bending something so that it deforms"
        3,  # "Bending something until it breaks"
        5,  # "Closing something"
        14,  # "Folding something"
        22,  # "Letting something roll along a flat surface"
        46,  # "Opening something":
        122,  # "Rolling something on a flat surface"
        134,  # "Something falling like a feather or paper"
        135,  # "Something falling like a rock"
        143,  # "Squeezing something"
        149,  # "Tearing something into two pieces"
        150,  # "Tearing something just a little bit"
        172,  # Unfolding something"
    ]  # np.arange(0, 174)

    ss2affordance = {
        2: 0,  # Bendable
        3: 0,
        5: 1,  # Openable/Closabe
        46: 1,
        14: 2,  # Foldable
        172: 2,
        22: 3,  # Rollable
        122: 3,
        149: 4,  # Tearable
        150: 4,
        143: 5,
        134: 6,  # Lightweight
        135: 7,  # Heavy
    }

    affordance_decoder = {
        0: "Bendable",
        1: "Openable/Closabe",
        2: "Foldable",
        3: "Rollable",  #
        4: "Tearable",
        5: "Squeezable",  #
        6: "Falling like a feather or paper",
        7: "Falling like a rock",
    }

    affordance_teacher_decoder = {
        "Bendable": 0,
        "Openable/Closabe": 2,
        "Foldable": 4,
        "Rollable": 6,
        "Tearable": 8,
        "Squeezable": 10,
        "Falling like a feather or paper": 12,
        "Falling like a rock": 13,
    }

    affordance_indices = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        13,
    ]
