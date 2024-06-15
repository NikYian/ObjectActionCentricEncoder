import torch.nn as nn


def get_criterion(args):
    if args.AcE_criterion == "MSE":
        return nn.MSELoss()
    elif args.AcE_criterion == "SmoothL1Loss":
        return nn.SmoothL1Loss(beta=0.2)
    else:
        print("Criterion not available")
        return None
