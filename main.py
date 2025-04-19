import argparse
import torch
from dataset import get_data_loaders, visualize_sample_from_loader
from Line_Detection import LaneSegmentationModel
from train import optimize, one_epoch_test
from optimization import get_optimizer, get_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Argument for line detection")
    # Add arguments
    parser.add_argument("--train", type=bool, help="Training model", default=False)
    parser.add_argument("--infer", type=bool, help="Inference model", default=False)

    return parser.parse_args()

def main():
    args = parse_args()

    # Common configuration
    num_classes = 2  # Number of classes, do not change this

    if args.train:
        # Hyperparameters for training
        batch_size = 128
        valid_size = 0.2
        num_epochs = 80
        opt = 'adamw'
        learning_rate = 0.0005
        dropout = 0.15
        weight_decay = 0.09

        # Init model
        model = LaneSegmentationModel(num_classes=num_classes, dropout=dropout)

        # Get dataloaders
        data_loaders = get_data_loaders(batch_size=batch_size, valid_size=valid_size)
        # visualize_sample_from_loader(data_loaders["train"], num_samples=3)

        # Optimizer and loss
        optimizer = get_optimizer(model=model, optimizer=opt,
                                  learning_rate=learning_rate, weight_decay=weight_decay)
        loss = get_loss()

        # Train the model
        optimize(
            data_loaders,
            model,
            optimizer,
            loss,
            n_epochs=num_epochs,
            save_path="checkpoints/best_val_loss.pt"
        )

    elif args.infer:
        # Load model for inference
        checkpoint_path = args.checkpoint or "checkpoints/best_val_loss.pt"
        model = LaneSegmentationModel(num_classes=num_classes)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        model.eval()

        # Run inference
        one_epoch_test(data_loaders['test'], model, loss)

    else:
        print("Please specify either --train or --infer")



if __name__ == '__main__':
    main()