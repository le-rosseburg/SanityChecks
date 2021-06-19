import torch, torchvision, statistics
import torch.nn as nn
import torchvision.transforms as transforms
from src.show_img import show_examples

use_subset = True  # Set this to True for debugging purposes.
if use_subset: print("Using subset...")

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
classes = train_dataset.classes

if use_subset:
    train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(0, 100))
    val_dataset = torch.utils.data.Subset(val_dataset, torch.arange(0, 100))

print(f'classes: {classes}\nnumber of instances:\n\ttrain: {len(train_dataset)}\n\tval: {len(val_dataset)}')

"""Visualizing a few examples:"""
show_examples(4, train_dataset, classes)

"""Creating dataloaders:"""
batch_size = 32
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

"""Defining the CNN:"""
net = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), padding=(1, 1)), # -> 48 activity maps with 32x32 image
    nn.ReLU(),
    nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding=(1, 1)), # -> 96 activity maps with 32x32 image
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # -> 96 activity maps with 16x16 image
    nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding=(1, 1)), # -> 192 activity maps with 16x16 image
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # -> 192 activity maps with 8x8 image
    nn.Flatten(), # -> vector with size 192*8*8
    nn.Linear(192 * 8 * 8, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# test the model on a single batch
# image_batch, target_batch = next(iter(train_dl))
# print(net(image_batch).shape)

"""Defining loss function and optimizer:"""
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)


"""Training:"""
use_gpu = True if torch.cuda.is_available() else False
print(f'Using cuda: {use_gpu}')
if use_gpu: net = net.cuda()

for epoch in range(5):  # loop over the dataset multiple times
    losses = []
    for inputs, targets in train_dl:
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()               # Reset gradients to zero
        outputs = net(inputs)               # Forward pass
        loss = loss_func(outputs, targets)  # Compute Loss
        loss.backward()                     # Compute the gradients
        optimizer.step()                    # Update parameter
        losses.append(loss.item())

    print(f"Epoch {epoch+1}: Current training loss {statistics.mean(losses)}")

print('Finished Training')


"""Validation:"""
with torch.no_grad():
    val_loss = 0.0
    correct, total = 0, 0
    for inputs, targets in val_dl:
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        val_loss += loss.item()
        predicted = outputs.argmax(1)  # select the class with the largest value

        total += len(targets)
        correct += (predicted == targets).sum().item()

print(f'Validation loss: {val_loss / len(val_dl)}')
print(f'Accuracy on the validation set: {100 * correct / total}%')