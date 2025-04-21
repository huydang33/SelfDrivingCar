import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce = self.ce(pred, target)
        dice = dice_loss(pred[:,1], target.float())  # dùng lớp lane (giả sử label 1 là lane)
        return ce + dice

def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True
    """
    loss  = CombinedLoss()

    return loss

def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    if optimizer.lower() == "sgd":
        # reate an instance of the SGD
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.SGD(
            model.parameters(),
            lr = learning_rate,
            momentum = momentum,
            weight_decay = weight_decay
        )

    elif optimizer.lower() == "adam":
        # Create an instance of the Adam optimizer.
        opt = torch.optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay
        )
    elif optimizer.lower() == "adamw":
        # Create an instance of the AdamW optimizer.
        opt = torch.optim.AdamW(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt