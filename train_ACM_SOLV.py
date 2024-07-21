# This code has been adapted and modified from  https://github.com/gorkaydemir/SOLV

import os
import time
import math
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from tqdm import tqdm
from args import Args

sys.path.append(r"./externals")
from SOLV.models.model import SOLV_nn, Visual_Encoder, MLP
from SOLV.read_args import get_args, print_args
import utils.SOLV_utils as utils

from models.AcE import ClassificationHead


def train_epoch(
    SOLV_args,
    AcE_args,
    vis_encoder,
    SOLV_model,
    optimizer,
    scheduler,
    train_dataloader,
    total_iter,
    evaluator,
    writer,
    AcE_model,
    ACM,
):
    vis_encoder.eval()
    SOLV_model.eval()
    AcE_model.eval()
    val_loader = tqdm(train_dataloader)

    bs = SOLV_args.batch_size
    H, W = SOLV_args.resize_to

    for i, (
        frames,
        input_masks,
        video_id,
        frame_idx,
        multi_label_aff,
        bb_masks,
    ) in enumerate(val_loader):
        H_t, W_t = frames.shape[-2:]

        frames = frames.cuda(non_blocking=True)  # (1, #frames + 2N, 3, H, W)
        input_masks = input_masks.cuda(non_blocking=True)  # (1, #frames + 2N)
        # bb = bb.cuda(non_blocking=True)
        bb_masks = bb_masks.cuda(non_blocking=True)
        multi_label_aff = multi_label_aff.cuda(non_blocking=True)

        # gt_masks = gt_masks.cuda(non_blocking=True)  # (1, #frames, H_t, W_t)

        with torch.cuda.amp.autocast(True):

            output_features, token_indices = vis_encoder(frames, get_gt=False)

            assert (
                output_features.isnan().any() == False
            ), f"{torch.sum(output_features.isnan())} items are NaN"

        output_features = output_features.to(torch.float32)

        reconstruction = SOLV_model(output_features, input_masks, token_indices)
        slots_AcE = AcE_model(reconstruction["target_frame_slots"].detach())
        new_slots, new_patch_attn, slot_nums = SOLV_model.merge(
            slots_AcE, reconstruction["attn"]
        )
        S = new_patch_attn.shape[1]
        masks = new_patch_attn.view(
            -1, S, H // SOLV_args.patch_size, W // SOLV_args.patch_size
        )  # (#frames, S, H // 8, W // 8)
        predictions = F.interpolate(
            masks, size=(H_t, W_t), mode="bilinear"
        )  # (#frames, S, H_t, W_t)
        predictions = torch.argmax(predictions, dim=1)  # (#frames, H_t, W_t)
        breakpoint()
    #     H_t, W_t = gt_masks.shape[-2:]
    #     frame_num = gt_masks.shape[1]

    #     H, W = SOLV_args.resize_to
    #     turn_number = model_input.shape[1] // bs
    #     if model_input.shape[1] % bs != 0:
    #         turn_number += 1

    #     # === DINO feature extraction ===
    #     all_dino_features = []
    #     all_token_indices = []
    #     for j in range(turn_number):
    #         s = j * bs
    #         e = (j + 1) * bs
    #         with torch.cuda.amp.autocast(True):
    #             features, token_indices = vis_encoder(
    #                 model_input[:, s:e], get_gt=True
    #             )  # (bs, token_num, 768), (bs, token_num)
    #             assert (
    #                 features.isnan().any() == False
    #             ), f"{torch.sum(features.isnan())} items are NaN"

    #         all_dino_features.append(features.to(torch.float32))
    #         all_token_indices.append(token_indices)

    #     all_dino_features = torch.cat(
    #         all_dino_features, dim=0
    #     )  # (#frames + 2N, token_num, 768)
    #     all_token_indices = torch.cat(
    #         all_token_indices, dim=0
    #     )  # (#frames + 2N, token_num)

    #     all_model_inputs = []
    #     all_model_tokens = []
    #     all_masks_input = []
    #     for i in range(frame_num):
    #         indices = list(range(i, i + (2 * SOLV_args.N + 1)))
    #         all_model_inputs.append(
    #             all_dino_features[indices].unsqueeze(dim=0)
    #         )  # (1, 2N + 1, token_num, 768)
    #         all_model_tokens.append(
    #             all_token_indices[indices].unsqueeze(dim=0)
    #         )  # (1, 2N + 1, token_num)
    #         all_masks_input.append(input_masks[:, indices])  # (1, 2N + 1)

    #     all_model_inputs = torch.cat(
    #         all_model_inputs, dim=0
    #     )  # (#frames, 2N + 1, token_num, 768)
    #     all_model_tokens = torch.cat(
    #         all_model_tokens, dim=0
    #     )  # (#frames, 2N + 1, token_num)
    #     all_masks_input = torch.cat(all_masks_input, dim=0)  # (#frames, 2N + 1)
    #     # === === ===

    #     turn_number = frame_num // bs
    #     if frame_num % bs != 0:
    #         turn_number += 1

    #     out_masks = []
    #     all_slots = []
    #     all_slot_nums = []
    #     for j in range(turn_number):
    #         s = j * bs
    #         e = (j + 1) * bs

    #         # === Input features ===
    #         features = all_model_inputs[s:e]  # (bs, 2N + 1, token_num, 768)
    #         features = torch.flatten(features, 0, 1)  # (bs * (2N + 1), token_num, 768)

    #         # === Token indices ===
    #         token_indices = all_model_tokens[s:e]  # (bs, 2N + 1, token_num)
    #         token_indices = torch.flatten(
    #             token_indices, 0, 1
    #         )  # (bs * (2N + 1), token_num)

    #         # === Attention masks ===
    #         input_masks_j = all_masks_input[s:e]

    #         reconstruction = SOLV_model(features, input_masks_j, token_indices)

    #         masks = reconstruction["mask"]  # (bs, S, token)
    #         slots = reconstruction["slots"]  # (bs, S, D_slot)
    #         slot_nums = reconstruction["slot_nums"]  # (bs)
    #         for l in range(slot_nums.shape[0]):
    #             slot_num = slot_nums[l]
    #             slots_l = slots[l, :slot_num]  # (S', D_slot)
    #             all_slots.append(slots_l)

    #         out_masks.append(masks)
    #         all_slot_nums.append(slot_nums)

    #     all_slots = torch.cat(all_slots, dim=0)  # (#slots, D_slot)
    #     all_slot_nums = torch.cat(all_slot_nums, dim=0)  # (#frames)
    #     masks = torch.cat(out_masks, dim=0)  # (#frames, S, token)

    #     S = masks.shape[1]

    #     masks = masks.view(
    #         -1, S, H // SOLV_args.patch_size, W // SOLV_args.patch_size
    #     )  # (#frames, S, H // 8, W // 8)
    #     predictions = F.interpolate(
    #         masks, size=(H_t, W_t), mode="bilinear"
    #     )  # (#frames, S, H_t, W_t)
    #     predictions = torch.argmax(predictions, dim=1)  # (#frames, H_t, W_t)

    #     if SOLV_args.merge_slots:
    #         predictions = SOLV.utils.bipartiate_match_video(
    #             all_slots, all_slot_nums, predictions
    #         )

    #     # === Instance Segmentation Evaluation ===
    #     miou = evaluator.update(predictions, gt_masks[0])
    #     loss_desc = f"mIoU: {miou * 100:.3f}"

    #     # === Logger ===
    #     val_loader.set_description(loss_desc)
    #     # === === ===

    # # === Evaluation Results ====
    # miou, fg_ari = evaluator.get_results()

    # # === Logger ===
    # print("\n=== Results ===")
    # print(f"\tmIoU: {miou * 100:.3f}")
    # print(f"\tFG-ARI: {fg_ari * 100:.3f}\n")

    # return miou, fg_ari


def main_worker(SOLV_args, AcE_args):

    print_args(SOLV_args)

    # === Dataloaders ====
    train_dataloader, val_dataloader, test_loader = utils.get_dataloaders(SOLV_args)
    # === Models ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vis_encoder = Visual_Encoder(SOLV_args).to(device)
    SOLV_model = SOLV_nn(SOLV_args).to(device)
    AcE_model = MLP(
        SOLV_args.slot_dim,
        4 * SOLV_args.slot_dim,
        SOLV_args.slot_dim,
        residual=True,
        layer_order="none",
    ).to(device)

    # === Misc ===
    evaluator = utils.Evaluator()
    writer = utils.get_writer(SOLV_args)

    # === Load from checkpoint ===
    to_restore = {"epoch": 0}

    utils.restart_from_checkpoint(
        SOLV_args.checkpoint_path,
        run_variables=to_restore,
        remove_module_from_key=True,
        model=SOLV_model,
    )

    checkpoint = torch.load(AcE_args.SOLV_AcE_checkpoint)
    msg = AcE_model.load_state_dict(checkpoint)
    print(msg)

    for param in SOLV_model.parameters():  # SOLV params are frozen
        param.requires_grad = False
    for param in AcE_model.parameters():  # SOLV params are frozen
        param.requires_grad = False

    ACM_model = ClassificationHead(SOLV_args.slot_dim)

    # === Training Items ===
    optimizer = torch.optim.Adam(
        utils.get_params_groups(ACM_model), lr=SOLV_args.learning_rate
    )
    scheduler = utils.get_scheduler(SOLV_args, optimizer, train_dataloader)

    ACM = {
        "model": ACM_model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    print("Starting ACM_SOLV training!")

    total_iter = 0
    best_val_acc = val_loss = 0

    for epoch in range(0, AcE_args.AcE_epochs):
        # train_dataloader.sampler.set_epoch(epoch)

        print(f"===== ===== [Epoch {epoch}] ===== =====")

        mean_loss, total_iter = train_epoch(
            SOLV_args,
            AcE_args,
            vis_encoder,
            SOLV_model,
            optimizer,
            scheduler,
            train_dataloader,
            total_iter,
            evaluator,
            writer,
            AcE_model,
            ACM,
        )

        val_loss = val_epoch(
            SOLV_args, vis_encoder, SOLV, val_dataloader, evaluator, writer, epoch
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"val loss: {val_loss} | best val loss: {best_val_loss}")

        path = os.path.join(AcE_args.log_dir, "SOLV_AcE.pth")
        torch.save(
            AcE_model.state_dict(),
            path,
        )

        # === Log ===
        writer.add_scalar("epoch/train-lsoss", mean_loss, epoch)
        writer.flush()
        writer.close()


if __name__ == "__main__":
    SOLV_args = get_args()
    AcE_args = Args()
    main_worker(SOLV_args, AcE_args)
