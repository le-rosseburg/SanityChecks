import torch
import matplotlib.pyplot as plt

def show_examples(n, train_dataset, classes):
    for i in range(n):
        index = torch.randint(0, len(train_dataset), size=(1,))  # select a random example
        image, target = train_dataset[index]
        # print(f'image of shape: {image.shape}')
        print(f'label: {classes[target]}')
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.show()