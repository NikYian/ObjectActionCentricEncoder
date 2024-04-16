from timm.models import create_model
import torch
import modeling_finetune
from collections import OrderedDict
import utils
import numpy as np
import torch.backends.cudnn as cudnn


def load_teacher(args):
    device = torch.device("cuda")
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    teacher = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )

    checkpoint = torch.load(args.finetune, map_location="cpu")
    print("Load ckpt from %s" % args.finetune)
    checkpoint_model = checkpoint["module"]
    state_dict = teacher.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith("backbone."):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith("encoder."):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict
    utils.load_state_dict(teacher, checkpoint_model, prefix=args.model_prefix)
    teacher.to(device)
    return teacher
