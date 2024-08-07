# This code has been adapted and modified from  https://github.com/gorkaydemir/SOLV

import os
import time
import math
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from args import Args


sys.path.append(r"./externals")
from SOLV.models.model import SOLV_nn, Visual_Encoder, MLP
from SOLV.read_args import get_args, print_args
import utils.SOLV_utils as utils
import utils.utils as AcE_utils

from models.AcE import ClassificationHead
from models.trainers.ACM_trainer import ACM_trainer
from segmentation_mask_overlay import overlay_masks


def train_epoch(
    SOLV_args,
    vis_encoder,
    SOLV_model,
    train_dataloader,
    total_iter,
    writer,
    AcE_model,
    ACM,
):
    total_loss = 0.0
    vis_encoder.eval()
    SOLV_model.eval()
    AcE_model.eval()
    ACM["model"].train()
    loader = tqdm(train_dataloader)

    bs = SOLV_args.batch_size
    H, W = SOLV_args.resize_to

    for iter, (
        frames,
        input_masks,
        video_id,
        frame_idx,
        multi_label_aff,
        bb_masks,
        target_frame_paths,
    ) in enumerate(loader):
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
        # slots_AcE = AcE_model(reconstruction["target_frame_slots"].detach())
        # new_slots, new_patch_attn, slot_nums = SOLV_model.merge(
        #     slots_AcE, reconstruction["sbind_attn"]
        # )
        new_slots, new_patch_attn, slot_nums = SOLV_model.merge(
            reconstruction["target_frame_slots"].detach(), reconstruction["sbind_attn"]
        )
        slots_AcE = AcE_model(new_slots)
        # new_slots, new_patch_attn, slot_nums = (
        #     reconstruction["slots"],
        #     reconstruction["attn"],
        #     reconstruction["slot_nums"],
        # )

        S = new_patch_attn.shape[1]
        masks = new_patch_attn.view(
            -1, S, H // SOLV_args.patch_size, W // SOLV_args.patch_size
        )  # (#frames, S, H // 8, W // 8)
        predictions = F.interpolate(
            masks, size=(H_t, W_t), mode="bilinear"
        )  # (#frames, S, H_t, W_t)
        predictions = torch.argmax(predictions, dim=1)  # (#frames, H_t, W_t)
        # find the slots of the interacting objects using the bounding box annotations
        slots_of_interacting_object = []
        for i in range(predictions.shape[0]):
            prediction = predictions[i] + 1
            prediction[bb_masks[i] == False] = 0
            prediction = torch.flatten(prediction)
            counts = torch.bincount(prediction)
            slot_of_interacting_object = torch.argmax(counts[1:]).item()
            slots_of_interacting_object.append(slot_of_interacting_object)

        labels = []
        inputs = []

        # concatenated_input = torch.cat((new_slots, new_patch_attn), dim=2)
        # concatenated_input = new_slots
        concatenated_input = torch.cat((new_slots, slots_AcE), dim=2)
        # concatenated_input = (new_slots + slots_AcE) / 2

        for i, slot_num in enumerate(slot_nums):
            non_object_slot_num = 1
            for j in range(slot_num):
                if j == slots_of_interacting_object[i]:
                    indices_to_remove = [0, 2]
                    mask = torch.ones(multi_label_aff[i].size(0), dtype=torch.bool)
                    mask[indices_to_remove] = False
                    ml_5 = multi_label_aff[i][mask]
                    labels.append(ml_5)
                    inputs.append(concatenated_input[i][j])
                elif non_object_slot_num > 0:
                    non_object_slot_num -= 1
                    labels.append(torch.zeros(5, dtype=int).cuda(non_blocking=True))
                    inputs.append(concatenated_input[i][j])
        labels = torch.stack(labels).detach()
        inputs = torch.stack(inputs).detach()
        aff_predictions = ACM["model"](inputs)

        ACM["optimizer"].zero_grad()
        # loss = F.mse_loss(aff_predictions, labels.float()).mean()
        loss = ACM["criterion"](aff_predictions, labels.float())
        total_loss += loss.item()
        loss.backward()
        ACM["optimizer"].step()
        ACM["scheduler"].step()
        lr = ACM["optimizer"].state_dict()["param_groups"][0]["lr"]
        mean_loss = total_loss / (iter + 1)
        loader.set_description(f"lr: {lr:.6f} | loss: {mean_loss:.5f}")

        writer.add_scalar("batch/loss", loss.item(), total_iter)

        total_iter += 1

    mean_loss = total_loss / (iter + 1)
    return mean_loss, total_iter


def val_epoch(
    SOLV_args,
    vis_encoder,
    SOLV_model,
    dataloader,
    total_iter,
    writer,
    AcE_model,
    ACM,
    test=False,
    ret="acc",
):
    total_loss = 0.0
    vis_encoder.eval()
    SOLV_model.eval()
    AcE_model.eval()
    ACM["model"].eval()
    loader = tqdm(dataloader)
    total_samples = 0
    y_pred_probs = []
    y_binary_list = []
    y_true = []

    bs = SOLV_args.batch_size
    H, W = SOLV_args.resize_to

    for iter, (
        frames,
        input_masks,
        video_id,
        frame_idx,
        multi_label_aff,
        bb_masks,
        target_frame_paths,
    ) in enumerate(loader):
        H_t, W_t = frames.shape[-2:]

        frames = frames.cuda(non_blocking=True)  # (1, #frames + 2N, 3, H, W)
        input_masks = input_masks.cuda(non_blocking=True)  # (1, #frames + 2N)
        # bb = bb.cuda(non_blocking=True)
        bb_masks = bb_masks.cuda(non_blocking=True)
        multi_label_aff_ = multi_label_aff.cuda(non_blocking=True)
        # multi_label_aff = torch.stack(multi_label_aff).transpose(0, 1)
        multi_label_aff = torch.where(
            multi_label_aff_ >= 1,
            torch.tensor(1, device=AcE_args.device),
            multi_label_aff_,
        ).cuda(non_blocking=True)

        # gt_masks = gt_masks.cuda(non_blocking=True)  # (1, #frames, H_t, W_t)

        with torch.cuda.amp.autocast(True):

            output_features, token_indices = vis_encoder(frames, get_gt=True)

            assert (
                output_features.isnan().any() == False
            ), f"{torch.sum(output_features.isnan())} items are NaN"

        output_features = output_features.to(torch.float32)

        reconstruction = SOLV_model(output_features, input_masks, token_indices)
        # slots_AcE = AcE_model(reconstruction["target_frame_slots"].detach())
        # new_slots, new_patch_attn, slot_nums = SOLV_model.merge(
        #     slots_AcE, reconstruction["sbind_attn"]
        # )
        new_slots, new_patch_attn, slot_nums = SOLV_model.merge(
            reconstruction["target_frame_slots"].detach(), reconstruction["sbind_attn"]
        )
        slots_AcE = AcE_model(new_slots)

        # new_slots, new_patch_attn, slot_nums = (
        #     reconstruction["slots"],
        #     reconstruction["attn"],
        #     reconstruction["slot_nums"],
        # )
        S = new_patch_attn.shape[1]
        masks = new_patch_attn.view(
            -1, S, H // SOLV_args.patch_size, W // SOLV_args.patch_size
        )  # (#frames, S, H // 8, W // 8)
        predictions = F.interpolate(
            masks, size=(H_t, W_t), mode="bilinear"
        )  # (#frames, S, H_t, W_t)
        predictions = torch.argmax(predictions, dim=1)  # (#frames, H_t, W_t)

        # find the slots of the interacting objects using the bounding box annotations
        slots_of_interacting_object = []
        for i in range(predictions.shape[0]):
            prediction = predictions[i] + 1
            prediction[bb_masks[i] == False] = 0
            prediction = torch.flatten(prediction)
            counts = torch.bincount(prediction)
            slot_of_interacting_object = torch.argmax(counts[1:]).item()
            slots_of_interacting_object.append(slot_of_interacting_object)

        labels = []
        inputs = []
        # concatenated_input = torch.cat((new_slots, new_patch_attn), dim=2)
        # concatenated_input = new_slots
        concatenated_input = torch.cat((new_slots, slots_AcE), dim=2)
        # concatenated_input = (new_slots + slots_AcE) / 2
        for i, slot_num in enumerate(slot_nums):
            for j in range(slot_num):
                if j == slots_of_interacting_object[i]:
                    indices_to_remove = [0, 2]
                    mask = torch.ones(multi_label_aff[i].size(0), dtype=torch.bool)
                    mask[indices_to_remove] = False
                    ml_5 = multi_label_aff[i][mask]
                    labels.append(ml_5)
                    inputs.append(concatenated_input[i][j])
                # else:
                #     labels.append(torch.zeros(5, dtype=int).cuda(non_blocking=True))
                #     inputs.append(concatenated_input[i][j])
        labels = torch.stack(labels).detach()
        inputs = torch.stack(inputs).detach()
        aff_predictions = ACM["model"](inputs).detach()
        binary_predictions = (aff_predictions > ACM["thresholds"]).int()

        y_pred_probs.append(aff_predictions.cpu().numpy())
        y_binary_list.append(binary_predictions.cpu().numpy())
        y_true.append(labels.cpu().numpy())

        if test:
            for i, path in enumerate(target_frame_paths):
                image = Image.open(path).convert("L")
                image = np.array(image)
                H_, W_ = image.shape
                predictions_ = F.interpolate(masks, size=(H_, W_), mode="bilinear")
                predictions_ = torch.argmax(predictions_, dim=1)
                prediction_ = predictions_[i].cpu().numpy()
                masks_, unique_values = utils.generate_boolean_masks(prediction_)
                mask_labels = [f"Slot_{i}" for i in range(masks_.shape[2])]
                cmap = plt.cm.tab20(np.arange(len(mask_labels)))[..., :-1]
                slots = concatenated_input[i][: len(unique_values)]
                aff_predictions = ACM["model"](slots).detach()
                aff_predictions = (
                    (aff_predictions > ACM["thresholds"]).int().cpu().numpy()
                )
                aff = AcE_args.affordances
                df = pd.DataFrame(aff_predictions, columns=aff).round(3).astype(str)
                df.index = mask_labels
                df.reset_index(inplace=True)
                df.rename(columns={"index": "Mask Labels"}, inplace=True)

                fig = overlay_masks(
                    image=image,
                    masks=masks_,
                    labels=mask_labels,
                    beta=0.2,
                    # colors=cmap,
                    mpl_figsize=(8, 8),
                    return_type="mpl",
                )
                fig_table = ff.create_table(df)
                fig_table.update_layout(
                    autosize=False, width=1000, height=200, font={"size": 9}
                )

                fig_table.write_image("table.png", scale=5)
                fig.savefig("test.png", dpi=300)
                plt.close()
                breakpoint()

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_binary_list = np.concatenate(y_binary_list, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    accuracy = np.mean(y_binary_list == y_true)

    if test:
        ACM["tresholds"] = ACM["trainer"].fine_tune_thresholds(
            y_true, y_pred_probs, num_classes=5
        )
        y_pred_probs = torch.tensor(y_pred_probs)
        y_binary_list = (y_pred_probs.cuda() > ACM["thresholds"]).int()
        y_binary_list = y_binary_list.cpu().numpy()

        ACM["trainer"].evaluate_multilabel_model(y_true, y_binary_list)
        # test masks

    else:
        ACM["tresholds"] = ACM["trainer"].fine_tune_thresholds(
            y_true, y_pred_probs, num_classes=5
        )
        print(ACM["tresholds"])

        y_pred_probs_rounded = np.round(y_pred_probs[:10], 3)
        print(y_pred_probs_rounded)
        print(y_true[:10])
        f1 = f1_score(y_true, y_binary_list, average="micro")
        print(f"f1 = {f1}")
        print(slot_nums)

    if ret == "f1":
        return f1_score(y_true, y_binary_list, average="micro")
    else:
        return accuracy, y_true, y_pred_probs, y_binary_list


def main_worker(SOLV_args, AcE_args):

    # print_args(SOLV_args)

    # === Dataloaders ====
    train_dataloader, val_dataloader, test_loader = utils.get_dataloaders(SOLV_args)
    # === Models ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vis_encoder = Visual_Encoder(SOLV_args).to(device)
    SOLV_model = SOLV_nn(SOLV_args).to(device)
    IO_classifier = MLP
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

    # ACM_model = ClassificationHead(
    #     992, num_classes=len(AcE_args.affordances), nn_type="hop"
    # ).to(device)

    # ACM_model = ClassificationHead(
    #     128, num_classes=len(AcE_args.affordances), nn_type="hop"
    # ).to(device)

    ACM_model = ClassificationHead(
        256, num_classes=len(AcE_args.affordances), nn_type="hop"
    ).to(device)

    checkpoint = torch.load("runs/SOLV_ACM_hop.pth")
    msg = ACM_model.load_state_dict(checkpoint)
    print(msg)
    # === Training Items ===
    optimizer = torch.optim.Adam(utils.get_params_groups(ACM_model), lr=AcE_args.AcE_lr)
    scheduler = utils.get_scheduler(AcE_args.AcE_epochs, optimizer, train_dataloader)
    trainer = ACM_trainer(
        AcE_args,
        None,
        None,
        None,
        None,
        None,
        None,
        AcE_args.device,
        AcE_args.log_dir,
    )
    ACM = {
        "model": ACM_model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": AcE_utils.get_criterion(AcE_args),
        "trainer": trainer,
        "thresholds": torch.tensor([0.5] * 5).to(AcE_args.device),
    }

    print("Starting ACM_SOLV training!")

    total_iter = 0
    best_val_acc = val_acc = 0

    # for epoch in range(0, AcE_args.AcE_epochs):
    #     # train_dataloader.sampler.set_epoch(epoch)

    #     print(f"===== ===== [Epoch {epoch}] ===== =====")

    #     mean_loss, total_iter = train_epoch(
    #         SOLV_args,
    #         vis_encoder,
    #         SOLV_model,
    #         train_dataloader,
    #         total_iter,
    #         writer,
    #         AcE_model,
    #         ACM,
    #     )
    #     if True:
    #         (val_acc, y_true, y_pred_probs, y_binary_list) = val_epoch(
    #             SOLV_args,
    #             vis_encoder,
    #             SOLV_model,
    #             val_dataloader,
    #             total_iter,
    #             writer,
    #             AcE_model,
    #             ACM,
    #         )
    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc

    #         print(f"val acc: {val_acc} | best val acc: {best_val_acc}")

    #     path = os.path.join(AcE_args.log_dir, "SOLV_ACM_hop.pth")
    #     torch.save(
    #         ACM["model"].state_dict(),
    #         path,
    #     )

    #     # === Log ===
    #     writer.add_scalar("epoch/train-lsoss", mean_loss, epoch)
    #     writer.flush()
    #     writer.close()

    (val_acc, y_true, y_pred_probs, y_binary_list) = val_epoch(
        SOLV_args,
        vis_encoder,
        SOLV_model,
        val_dataloader,
        total_iter,
        writer,
        AcE_model,
        ACM,
    )

    (accuracy, y_true, y_pred_probs, y_binary_list) = val_epoch(
        SOLV_args,
        vis_encoder,
        SOLV_model,
        test_loader,
        total_iter,
        writer,
        AcE_model,
        ACM,
        test=True,
    )


if __name__ == "__main__":
    SOLV_args = get_args()
    AcE_args = Args()
    main_worker(SOLV_args, AcE_args)
