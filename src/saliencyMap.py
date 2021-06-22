import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import torch.nn.functional as F

# maps val ranging from start1 to stop1 to new range start2 to stop2
def normValue(val, start1, stop1, start2, stop2):
    return (val - start1) / (stop1 - start1) * (stop2 - start2) + start2

def imshow(img, cmap=None, title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=cmap)

def transform(img):
    return np.transpose(img.cpu().detach().numpy(), (1, 2, 0))

def main(net, val_dl, classes):
    print("\n\n#####################\n### Saliency Maps ###\n#####################")
    net.eval()

    dataiter = iter(val_dl)
    images, labels = next(dataiter)
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    """Plot samples and the prediction of it"""
    predict_samples(net, images, labels, classes, n=6)

    """Plot saliency map:"""
    index = 3
    outputs = net(images[index].unsqueeze(0))
    predicted = outputs.argmax(1)
    print('Predicted:', classes[predicted], ' Probability:', torch.max(F.softmax(outputs, 1)).item())

    ig = IntegratedGradients(net)
    attr_ig = ig.attribute(images[index].unsqueeze(0), target=labels[index])
    attr_ig = transform(attr_ig.squeeze())
    attr_ig_sum = attr_ig.sum(2)
    attr_ig_sum_normed = normValue(attr_ig_sum, np.min(attr_ig_sum), np.max(attr_ig_sum) , 0, 1)

    imshow(transform(images[index]), title="Original Image")
    imshow(attr_ig_sum_normed, cmap="gray", title="Grayed Saliency Map")
    viz.visualize_image_attr(attr_ig, (transform(images[index]))/2 + 0.5, method="blended_heat_map",sign="all", show_colorbar=True, title="Overlayed Integrated Gradients")
    plt.show()

def predict_samples(net, images, labels, classes, n=6):
    outputs = net(images[:n])
    predicted = outputs.argmax(1)
    for i in range(n):
        print(f"IMG{i}: Actual {classes[labels[i]].ljust(10,' ')} Predicted {classes[predicted[i]]}")
    print()
    imshow(transform(torchvision.utils.make_grid(images[:6])))
    plt.show()