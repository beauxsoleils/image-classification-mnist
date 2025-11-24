#!/usr/bin/env python3

"""
The main() script for initializing our model and running training epochs.

"""

import torch, utils

from torch import optim, nn
from model import Model
from mnist import get_dataloaders_mnist
from train import run_training_loop, evaluate
from args import build_interface

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

def main():
    
    # Build ui
    args: ArgumentParser = build_interface()

    # Select chipset 
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Prepare datasets
    training_loader, testing_loader = get_dataloaders_mnist(
        batch_size=args.batch_size
    )

    # Define model
    model = Model()

    # Initialize loss function
    loss_function = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = optim.SGD(
        params=model.parameters(), 
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    images, labels = next(iter(training_loader))
    grid = make_grid(images)

    # init SummaryWriter
    tb = SummaryWriter() 
    tb.add_image("images", grid)
    tb.add_graph(model, images)
    
    # Begin training 
    print("Training Launched")
    print(f"Device: {device}\n")

    final_train_loss: float = run_training_loop(
        model=model, 
        loss_fn=loss_function, 
        optimizer=optimizer, 
        training_loader=training_loader, 
        testing_loader=testing_loader,
        device=device,
        epochs=args.epochs,
        tb=tb
    )

    print(f"Final Loss: {final_train_loss}\n")

    if args.save:
        checkpoint_path = utils.save_model_state(model)
        print(f"Model weights saved to /{checkpoint_path}\n")

    tb.close()

if __name__ == "__main__": main()



