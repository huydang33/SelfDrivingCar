import argparse
import torch
from dataset import get_data_loaders
from Line_Detection import LaneSegmentationModel
from train import optimize, one_epoch_test
from optimization import get_optimizer, get_loss
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Argument for line detection")
    # Add arguments
    parser.add_argument("--train", type=bool, help="Training model", default=False)
    parser.add_argument("--resume", type=bool, help="Resume training", default=False)
    parser.add_argument("--infer", type=bool, help="Inference model", default=False)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=False)
    parser.add_argument("--kaggle", type=bool, help="using kaggle", default=False)
    parser.add_argument("--data_dir", type=str, help="data_dir", default="data/")

    return parser.parse_args()

def main():
    args = parse_args()

    # Common configuration
    num_classes = 2  # Number of classes, do not change this

    if args.train:
        # Hyperparameters for training
        batch_size = args.batch_size if args.batch_size else 16  # Default batch size
        valid_size = 0.2
        num_epochs = 80
        opt = 'adamw'
        learning_rate = 0.0005
        dropout = 0.15
        weight_decay = 0.09

        # Init model
        model = LaneSegmentationModel(num_classes=num_classes, dropout=dropout)

        # Get dataloaders
        if args.kaggle:
            import kagglehub
            culane_root = kagglehub.dataset_download('manideep1108/culane')
        else:
            culane_root = args.data_dir
            # check if the path exists
            if not os.path.exists(culane_root):
                raise FileNotFoundError(f"Data directory {culane_root} does not exist.")
        data_loaders = get_data_loaders(culane_root, batch_size=batch_size, valid_size=valid_size)
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
            save_path="checkpoints/best_val_loss.pt",
            resume=args.resume
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