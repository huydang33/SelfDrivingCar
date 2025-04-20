import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
import os


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """

    if torch.cuda.is_available():
        # Transfer the model to the GPU
        model.cuda()

    # Set the model to training mode
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Forward pass: compute predicted outputs by passing inputs to the model
        output  = model(data)
        # Calculate the loss
        loss_value  = loss(output, target.squeeze(1).long())
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()
        # Perform a single optimization step (parameter update)
        optimizer.step()

        # Update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():

        # Set the model to evaluation mode
        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # Move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Forward pass: compute predicted outputs by passing inputs to the model
            output  = model(data)
            # Calculate the loss
            loss_value  = loss(output, target.squeeze(1).long())

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss_fn, n_epochs, save_path, resume=False):
    start_epoch = 1
    valid_loss_min = None    
    trigger_times = 0
    patience = 5  # Early stopping patience

    liveloss = PlotLosses()

    # Resume từ checkpoint nếu có
    if resume and os.path.exists(save_path):
        print("🔄 Resuming from checkpoint:", save_path)
        checkpoint = torch.load(save_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint)
        start_epoch = 7
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(start_epoch, n_epochs + 1):
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss_fn)
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss_fn)

        logs = {
            'log loss': train_loss,
            'val_log loss': valid_loss,
        }
        liveloss.update(logs)
        liveloss.send()

        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}")

        if valid_loss_min is None or ((valid_loss_min - valid_loss) / valid_loss_min > 0.01):
            print(f"✅ New minimum validation loss: {valid_loss:.6f}. Saving model ...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_loss_min": valid_loss
            }, save_path)
            valid_loss_min = valid_loss
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"⚠️ No improvement. Trigger count: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("🛑 Early stopping triggered.")
                break

        scheduler.step(valid_loss)


def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits  = model(data)
            # 2. calculate the loss
            loss_value  = loss(logits, target)

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            _, pred  = torch.max(logits.data, 1)

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss