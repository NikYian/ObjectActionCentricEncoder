import torch.nn as nn
import math
import torch
from PIL import Image, ImageDraw, ImageFont


def get_criterion(args):
    if args.AcE_criterion == "MSE":
        return nn.MSELoss()
    elif args.AcE_criterion == "SmoothL1Loss":
        return nn.SmoothL1Loss(beta=1)
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


def add_text_to_image(image_path, output_path, text_dict):
    image = Image.open(image_path)
    image_width, image_height = image.size

    text_width = 150  # Width of the text area
    new_width = image_width + text_width
    new_height = max(image_height, 150)
    new_image = Image.new("RGB", (new_width, 600), (255, 255, 255))

    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)

    font = ImageFont.load_default(10)

    text = "\n".join([f"{k}: {v:.2f}" for k, v in text_dict.items()])

    text_position = (image_width + 10, 10)  # 10 pixels padding from the image

    draw.text(text_position, text, fill="black", font=font)

    if output_path:
        new_image.save(output_path)

    return new_image


def create_image_grid(image_paths, predictions, output_path, grid_size=(5, 3)):
    """Create a grid of images with text."""
    font = ImageFont.load_default()

    sample_image = Image.open(image_paths[0])
    image_width, image_height = 300, 300
    text_width = 300  # Width of the text area
    total_width = (image_width + text_width) * grid_size[1]
    total_height = image_height * grid_size[0]

    grid_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    for idx, (image_path, prediction) in enumerate(zip(image_paths, predictions)):
        # image = Image.open(image_path)

        # text = "\n".join([f"{k}: {v:.2f}" for k, v in prediction.items()])

        image_with_text = add_text_to_image(image_path, None, prediction)

        row = idx // grid_size[1]
        col = idx % grid_size[1]
        x = col * (image_width + text_width)
        y = row * image_height

        grid_image.paste(image_with_text, (x, y))

    grid_image.save(output_path)
