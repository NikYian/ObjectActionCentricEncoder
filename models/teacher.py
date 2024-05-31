from timm.models import create_model
import torch
from collections import OrderedDict
import externals.VideoMAE.utils as utils
import numpy as np
import torch.backends.cudnn as cudnn
from externals.VideoMAE import modeling_finetune
from torch import nn
from torch.nn.functional import softmax


def load_teacher(args, aff_head=False):
    device = args.device
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

    if aff_head:
        teacher = TeacherWtAffHead(teacher, args)
    teacher.to(device)
    return teacher


class TeacherWtAffHead(nn.Module):
    def __init__(self, teacher, args):
        super(TeacherWtAffHead, self).__init__()
        self.args = args

        self.VAE = teacher
        for param in self.VAE.parameters():  # CLIP params are frozen
            param.requires_grad = False

        output_sice = (
            len(args.affordance_decoder) * 2
        ) - 2  # all affordances have a "negative" apart from the two last options

        self.head = nn.Linear(args.AcE_feature_size, output_sice)

        if args.teacher_head_checkpoint:
            checkpoint = torch.load(args.teacher_head_checkpoint)
            self.head.load_state_dict(checkpoint)

    def forward(self, clips):
        VAE_features = self.VAE.forward_features(clips)
        affordance_logits = self.head(VAE_features)

        # apply softmax in pairs of 2
        reshaped_tensor = affordance_logits.view(-1, 2)
        softmaxed_pairs = softmax(reshaped_tensor, dim=1)
        output = softmaxed_pairs.view(affordance_logits.size())
        return output

    def aff(self, clips):
        self.eval()
        batch_size = clips.shape[0]

        res = self.forward(clips).cpu().detach().numpy().astype(np.float32)
        res_dicts = []
        for clip_res in res:
            dict = {}
            for aff, indx in self.args.affordance_teacher_decoder.items():
                dict[aff] = clip_res[indx]
            res_dicts.append(dict)

        return res
        # res = res.view(-1, 2)
        # res  = res[
