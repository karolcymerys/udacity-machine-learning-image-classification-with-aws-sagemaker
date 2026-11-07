import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.profiler import profile, ProfilerActivity, record_function, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

global_step = 0

def test(model, test_loader, criterion):
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    losses = []
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        losses.append(loss.item())

    return (
        np.mean(losses),
        correct / total
    )


def train(model, train_loader, criterion, optimizer, profiler, tensorboard_writer):
    global global_step
    model.train()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    losses = []
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with record_function("train_forward"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        with record_function("train_backward"):
            loss.backward()

        with record_function("train_optimizer_step"):
            optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels.data)
        accuracy = corrects.item() / inputs.size(0)
        logger.info(f'Train Batch {idx}: Loss: {loss.item()}, Accuracy: {accuracy:.8f}')

        correct += corrects.item()
        total += labels.size(0)
        losses.append(loss.item())

        tensorboard_writer.add_scalar("Batch/Loss", loss.item(), global_step)
        tensorboard_writer.add_scalar("Batch/Accuracy", accuracy, global_step)
        global_step += 1

        profiler.step()

    return (
        model,
        np.mean(losses),
        correct / total
    )


def net(num_classes, device):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(device)


def create_data_loaders(train_data_path,
                        validation_data_path,
                        test_data_path,
                        batch_size):
    # Create transforms for the training and testing data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)), # ResNet50 expects 224x224 images, crop randomly for data augmentation
        transforms.RandomHorizontalFlip(p=0.5),   # Randomly flip horizontally for data augmentation
        transforms.ColorJitter(                   # Slight color variations to make model resilient to lighting conditions
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    )


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initiate the model
    model = net(args.num_classes, device)

    # Prepare the data loaders
    train_loader, validation_loader, test_loader = create_data_loaders(
        args.train_data_dir,
        args.validation_data_dir,
        args.test_data_dir,
        args.batch_size
    )

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        schedule=schedule(
            wait=2,
            warmup=1,
            active=8,
            repeat=1
        ),
        on_trace_ready=tensorboard_trace_handler(
            os.path.join(args.output_dir, 'tensorboard', 'profiler'),
        ),
        record_shapes=True,
        profile_memory=True
    ) as profiler:

        # Define loss function
        loss_criterion = torch.nn.CrossEntropyLoss()

        # Define optimizer
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        logger.info(
            "Starting training with the following parameters:\n"
            f"Batch size: {args.batch_size}\n"
            f"Epochs: {args.epochs}\n"
            f"Learning rate: {args.lr}\n"
            f"Optimizer: {args.optimizer}\n"
        )

        tensorboard_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard', 'metrics'))

        # Training loop
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            model, train_avg_loss, train_accuracy = train(model, train_loader, loss_criterion, optimizer, profiler, tensorboard_writer)
            logger.info(f'Epoch [{epoch + 1}]\tTrain Avg Loss: {train_avg_loss:.8f}\tTrain Accuracy: {train_accuracy:.4f}')
            tensorboard_writer.add_scalar("Train/Loss", train_avg_loss, epoch + 1)
            tensorboard_writer.add_scalar("Train/Accuracy", train_accuracy, epoch + 1)

            val_avg_loss, val_accuracy = test(model, validation_loader, loss_criterion)
            logger.info(f'Epoch [{epoch + 1}]\tValidation Avg Loss: {val_avg_loss:.8f}\tValidation Accuracy: {val_accuracy:.4f}')
            tensorboard_writer.add_scalar("Validation/Loss", val_avg_loss, epoch + 1)
            tensorboard_writer.add_scalar("Validation/Accuracy", val_accuracy, epoch + 1)

        # Test the model
        test_avg_loss, test_accuracy = test(model, test_loader, loss_criterion)
        logger.info(f'Test Avg Loss: {test_avg_loss:.8f}\tTest Accuracy: {test_accuracy:.4f}')

        # Save the trained model
        torch.save(model, os.path.join(args.model_dir, "model.pth"))

        tensorboard_writer.flush()
        tensorboard_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='adam',
        help="Optimizer (default: Adam). Supported: 'sgd', 'adam', 'rmsprop'"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of classes in the dataset",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        "--validation-data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_VALIDATION"]
    )
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_TEST"]
    )
    args = parser.parse_args()

    main(args)
