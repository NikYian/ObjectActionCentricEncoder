import argparse
import torch


class Args(argparse.Namespace):
    batch_size = 1
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
    data_path = "/gpu-data2/nyian/ssv2/mp4"
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
    something_else_ann = [
        "ssv2/something_else/bounding_box_smthsmth_part1.json",
        "ssv2/something_else/bounding_box_smthsmth_part2.json",
        "ssv2/something_else/bounding_box_smthsmth_part3.json",
        "ssv2/something_else/bounding_box_smthsmth_part4.json",
    ]

    obj_crop_dir = "/gpu-data2/nyian/ssv2/object_crops"
    image_featrures_dir = "/gpu-data2/nyian/ssv2/mae_features"  # "ssv2/CLIP_features" or "ssv2/mae_features"
    VAE_features_dir = "/gpu-data2/nyian/ssv2/VAE_features"
    sa_sample_ids = {
        "train": "ssv2/somethings_affordances/train_comp.json",
        "val": "ssv2/somethings_affordances/val_comp.json",
        "test": "ssv2/somethings_affordances/test_comp.json",
    }

    # AcE arguments
    image_features = "mae"  # "mae" or "clip"
    AcE_epochs = 30
    head = "MLP"  # "MLP" or "Hopfield"
    AcE_checkpoint = "runs/AcE_MLP.pth"
    SOLV_AcE_checkpoint = "runs/SOLV_AcE.pth"
    ACM_type = "Hopfield"  # "MLP" or "Hopfield"
    ACM_checkpoint = None  # "runs/ACM_Hopfield_combo.pth"
    ACM_features = "combo"  # "image", "AcE", "combo"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLIP_model = "ViT-B/32"
    split_ratios = [0.4, 0.3, 0.3]
    AcE_batch_size = 1000
    AcE_dropout_rate = 0.1
    AcE_feature_size = 384
    AcE_hidden_size = 1024
    AcE_hidden_layers = [1024]
    # AcE_hidden_layers = [
    #     500,
    #     600,
    #     700,
    #     800,
    #     900,
    #     1000,
    #     900,
    #     800,
    #     700,
    #     600,
    #     500,
    #     400,
    #     300,
    # ]
    AcE_criterion = "MSE"  # "MSE" or "SmoothL1Loss"
    AcE_lr = 1e-4
    AcE_weight_decay = 10e-6

    teacher_head_checkpoint = "checkpoints/teacher_head_4.pth"
    teacher_lr = 10e-5
    teacher_epochs = 5
    temperature_init = 0.2

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

    # ssv2 labels and the coresponding affordances
    action2aff_labels = {
        # [ss class, object that is characterized by the action, affordance]
        2: ["Bending something so that it deforms", "object 0", "bendable"],
        3: ["Bending something until it breaks", "object 0", "bendable"],
        4: ["Burying something in something", "object 1", "burry in/cover with"],
        5: ["Closing something", "object 0", "openable/closable"],
        6: ["Covering something with something", "object 1", "burry in/cover with"],
        7: ["Digging something out of something", "object 1", "burry in/cover with"],
        14: ["Folding something", "object 0", "foldable"],
        22: ["Letting something roll along a flat surface", "object 0", "rollable"],
        23: ["Letting something roll down a slanted surface", "object 0", "rollable"],
        24: [
            "Letting something roll up a slanted surface, so it rolls back down",
            "object 0",
            "rollable",
        ],
        46: ["Opening something", "object 0", "openable/closable"],
        49: ["Plugging something into something", "object 1", "plug into"],
        50: [
            "Plugging something into something but pulling it right out as you remove your hand",
            "object 1",
            "plug into",
        ],
        # 51: ["Poking a hole into some substance", "object 0", "poke hole"],
        # 52: ["Poking a hole into something soft", "object 0", "poke hole"],
        59: ["Pouring something into something", "object 1", "containment"],
        60: [
            "Pouring something into something until it overflows",
            "object 1",
            "containment",
        ],
        62: ["Pouring something out of something", "object 1", "containment"],
        66: [
            "Pretending to close something without actually closing it",
            "object 0",
            "openable/closable",
        ],
        67: [
            "Pretending to open something without actually opening it",
            "object 0",
            "openable/closable",
        ],
        70: [
            "Pretending to pour something out of something, but something is empty",
            "object 1",
            "containment",
        ],
        # 91: [
        #     "Pulling two ends of something so that it gets stretched",
        #     "object 0",
        #     "stretchable",
        # ],
        # 92: [
        #     "Pulling two ends of something so that it separates into two pieces",
        #     "object 0",
        #     "pull ends to separate",
        # ],
        115: [
            "Putting something that can't roll onto a slanted surface, so it slides down",
            "object 0",
            "can't roll/slide",
        ],
        116: [
            "Putting something that can't roll onto a slanted surface, so it stays where it is",
            "object 0",
            "can't roll/slide",
        ],
        122: [
            "Rolling something on a flat surface",
            "object 0",
            "rollable",
        ],
        # 123: ["Scooping something up with something", "object 1", "scoop with"],
        129: ["Showing that something is empty", "object 0", "containment"],
        # 134: [
        #     "Something falling like a feather or paper",
        #     "object 0",
        #     "falls like a feather or paper",
        # ],
        # 135: [
        #     "Something falling like a rock",
        #     "object 0",
        #     "falls like a rock",
        # ],
        143: ["Squeezing something", "object 0", "squeezable"],
        # 149: ["Tearing something into two pieces", "object 0", "tearable"],
        150: ["Tearing something just a little bit", "object 0", "tearable"],
        # 162: [
        #     "Trying to bend something unbendable so nothing happens",
        #     "object 0",
        #     "unbendable",
        # ],
        172: ["Unfolding something", "object 0", "foldable"],
    }

    affordances = [
        "bendable",
        "foldable",
        "openable/closable",
        "rollable",
        "can't roll/slide",
        "squeezable",
        "containment",
        "tearable",
        "plug into",
        "burry in/cover with",
    ]
