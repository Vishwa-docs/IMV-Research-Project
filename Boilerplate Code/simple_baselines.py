import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import numpy as np
import os

def parse_args():
    """
    Parse input arguments
    Returns
    -------
    args : object
        Parsed args
    """
    h = {
        "program": "Simple Baselines training",
        "train_folder": "Path to training data folder.",
        "batch_size": "Number of images to load per batch. Set according to your PC GPU memory available. If you get "
                      "out-of-memory errors, lower the value. defaults to 64",
        "epochs": "How many epochs to train for. Once every training image has been shown to the CNN once, an epoch "
                  "has passed. Defaults to 15",
        "test_folder": "Path to test data folder",
        "num_workers": "Number of workers to load in batches of data. Change according to GPU usage",
        "test_only": "Set to true if you want to test a loaded model. Make sure to pass in model path",
        "model_path": "Path to your model",
        "learning_rate": "The learning rate of your model. Tune it if it's overfitting or not learning enough"}
    parser = argparse.ArgumentParser(description=h['program'], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_folder', help=h["train_folder"], type=str)
    parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=64)
    parser.add_argument('--epochs', help=h["epochs"], type=int, default=5)
    parser.add_argument('--test_folder', help=h["test_folder"], type=str)
    parser.add_argument('--num_workers', help=h["num_workers"], type=int, default=5)
    parser.add_argument('--test_only', help=h["test_only"], type=bool, default=False)
    parser.add_argument('--model_path', help=h["num_workers"], type=str),
    parser.add_argument('--learning_rate', help=h["learning_rate"], type=float, default=0.003)

    args = parser.parse_args()

    return args


def load_train_data(train_data_path, batch_size):
    # Convert images to tensors, normalize, and resize them
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    return train_data_loader


def load_test_data(test_data_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
    names = test_data.imgs
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    return test_data_loader, names


def train():
    args = parse_args()
    train_data = load_train_data(args.train_folder, args.batch_size)
    train_losses = []

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    my_net = nn.Sequential(nn.Linear(1000,10))
    model = nn.Sequential(models.resnet18(pretrained=True), my_net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    model.to(device)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        with tqdm(train_data, unit="batch") as tepoch:
            for inputs, labels in tepoch:

                tepoch.set_description(f"Epoch {epoch}")
                # get the inputs
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + get predictions + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == labels).sum().item()
                accuracy = correct / args.batch_size

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)


    print('Finished Training')
    torch.save(model, 'unicornmodel.pth')

def test(model_path):
    args = parse_args()
    test_data, files = load_test_data(args.test_folder, args.batch_size)
    files = np.array(files)[:,0]
    names = []
    for file in files:
        _, name = os.path.split(file)
        names += [name[4:-4]]
    names = np.array(names)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = torch.load(args.model_path)
    model.to(device)
    network_output = np.array([])
    scores = np.array([])
    correct = 0
    total = 0
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'class_id', 'score'])
        with torch.no_grad():
            for data in test_data:
                images, _ = data
                images = images.to(device)
                outputs = model(images)
                score, predicted = torch.max(outputs.data, 1)

                network_output = np.concatenate((network_output,predicted.detach().cpu().numpy()))
                scores = np.concatenate((scores,score.detach().cpu().numpy()))

        for idx, name in enumerate(names):
            writer.writerow([name, int(network_output[idx]), scores[idx]])




if __name__ == "__main__":
    args = parse_args()
    if args.test_only:
        test(args.model_path)
    else:
        train()
