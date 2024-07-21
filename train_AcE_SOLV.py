# This code has been adapted and modified from  https://github.com/gorkaydemir/SOLV

import os
import time
import math
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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


def train_epoch(
    SOLV_args,
    AcE_args,
    vis_encoder,
    SOLV,
    optimizer,
    scheduler,
    train_dataloader,
    total_iter,
    writer,
    AcE,
):
    total_loss = 0.0
    vis_encoder.eval()
    SOLV.eval()

    loader = tqdm(train_dataloader)

    for i, (
        frames,
        masks,
        video_id,
        frame_idx,
        multi_label_aff,
        bb,
    ) in enumerate(loader):

        frames = frames.cuda(non_blocking=True)  # (B, F, 3, H, W)
        masks = masks.cuda(non_blocking=True)  # (B, F)

        B = frames.shape[0]
        with torch.cuda.amp.autocast(True):
            output_features, _ = vis_encoder(frames[:, [SOLV_args.N]], get_gt=True)
            dropped_features, token_indices = vis_encoder(frames, get_gt=False)

            assert (
                output_features.isnan().any() == False
            ), f"{torch.sum(output_features.isnan())} items are NaN"
            assert (
                dropped_features.isnan().any() == False
            ), f"{torch.sum(dropped_features.isnan())} items are NaN"

        output_features = output_features.to(torch.float32)
        dropped_features = dropped_features.to(torch.float32)

        reconstruction = SOLV(dropped_features, masks, token_indices)

        # associative memory train
        AcE["optimizer"].zero_grad()
        predictions = AcE["model"](reconstruction["target_frame_slots"].detach())
        loss = F.mse_loss(predictions, reconstruction["slots_temp"].detach()).mean()
        total_loss += loss.item()
        loss.backward()
        AcE["optimizer"].step()
        AcE["scheduler"].step()

        # if args.gpu == 0:
        lr = AcE["optimizer"].state_dict()["param_groups"][0]["lr"]
        mean_loss = total_loss / (i + 1)
        loader.set_description(f"lr: {lr:.6f} | loss: {mean_loss:.5f}")

        writer.add_scalar("batch/loss", loss.item(), total_iter)

        total_iter += 1

    mean_loss = total_loss / (i + 1)
    return mean_loss, total_iter


@torch.no_grad()
def val_epoch(
    SOLV_args,
    AcE_args,
    vis_encoder,
    SOLV,
    val_dataloader,
    evaluator,
    writer,
    epoch,
    AcE,
):
    vis_encoder.eval()
    SOLV.eval()
    AcE["model"].eval()
    val_loader = tqdm(val_dataloader)

    total_loss = 0.0

    for i, (
        frames,
        input_masks,
        video_id,
        frame_idx,
        multi_label_aff,
        bb,
    ) in enumerate(val_loader):

        frames = frames.cuda(non_blocking=True)  # (1, #frames + 2N, 3, H, W)
        masks = input_masks.cuda(non_blocking=True)  # (1, #frames + 2N)

        B = frames.shape[0]
        with torch.cuda.amp.autocast(True):
            output_features, _ = vis_encoder(frames[:, [SOLV_args.N]], get_gt=True)
            dropped_features, token_indices = vis_encoder(frames, get_gt=False)

            assert (
                output_features.isnan().any() == False
            ), f"{torch.sum(output_features.isnan())} items are NaN"
            assert (
                dropped_features.isnan().any() == False
            ), f"{torch.sum(dropped_features.isnan())} items are NaN"

        output_features = output_features.to(torch.float32)
        dropped_features = dropped_features.to(torch.float32)

        reconstruction = SOLV(dropped_features, masks, token_indices)
        predictions = AcE["model"](reconstruction["target_frame_slots"].detach())
        loss = F.mse_loss(predictions, reconstruction["slots_temp"].detach()).mean()
        total_loss += loss.item()

    mean_loss = total_loss / (i + 1)
    return mean_loss


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

    # === Training Items ===
    optimizer = torch.optim.Adam(
        SOLV.utils.get_params_groups(SOLV_model), lr=SOLV_args.learning_rate
    )
    scheduler = SOLV.utils.get_scheduler(SOLV_args, optimizer, train_dataloader)

    AcE_optimizer = torch.optim.Adam(
        AcE_model.parameters(), lr=SOLV_args.mem_learning_rate
    )
    AcE_scheduler = SOLV.utils.get_scheduler(SOLV_args, AcE_optimizer, train_dataloader)
    AcE = {
        "model": AcE_model,
        "optimizer": AcE_optimizer,
        "scheduler": AcE_scheduler,
    }
    # === Misc ===
    evaluator = SOLV.utils.Evaluator()
    writer = SOLV.utils.get_writer(SOLV_args)

    print(f"Loss, optimizer and schedulers ready.")

    # === Load from checkpoint ===
    to_restore = {"epoch": 0}
    if SOLV_args.use_checkpoint:
        SOLV.utils.restart_from_checkpoint(
            SOLV_args.checkpoint_path,
            run_variables=to_restore,
            remove_module_from_key=True,
            model=SOLV_model,
        )
    for param in SOLV_model.parameters():  # SOLV params are frozen
        param.requires_grad = False

    print("Starting AcE_SOLV training!")

    total_iter = 0
    best_val_loss = val_loss = float("inf")

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
            writer,
            AcE,
        )

        val_loss = val_epoch(
            SOLV_args,
            AcE_args,
            vis_encoder,
            SOLV_model,
            val_dataloader,
            evaluator,
            writer,
            epoch,
            AcE,
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
