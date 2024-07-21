import torch.nn as nn
import math
import torch


def get_criterion(args):
    if args.AcE_criterion == "MSE":
        return nn.MSELoss()
    elif args.AcE_criterion == "SmoothL1Loss":
        return nn.SmoothL1Loss(beta=2)
        # return nn.MSELoss()
    else:
        print("Criterion not available")
        return None


# get_scheduler taken from https://github.com/gorkaydemir/SOLV/blob/main/utils.py
def get_scheduler(args, optimizer, train_loader):
    T_max = len(train_loader) * args.AcE_epochs
    warmup_steps = int(T_max * 0.05)
    steps = T_max - warmup_steps
    gamma = math.exp(math.log(0.5) / (steps // 3))

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, total_iters=warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear_scheduler, scheduler],
        milestones=[warmup_steps],
    )
    return scheduler
