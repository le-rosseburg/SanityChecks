import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz


def transform(img):
    return np.transpose(img.cpu().detach().numpy(), (1, 2, 0))

def main(net, val_dl, index):
    net.eval()

    images, labels = next(iter(val_dl))
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    """Plot saliency map:"""
    ig = IntegratedGradients(net)
    attr_ig = ig.attribute(images[index].unsqueeze(0), target=labels[index])
    attr_ig = transform(attr_ig.squeeze())

    original_img = transform(images[index])
    viz.visualize_image_attr(None, original_img, method="original_image", title="Original Image")
    viz.visualize_image_attr(attr_ig, original_img / 2 + 0.5, method="blended_heat_map",sign="all", show_colorbar=True, title="Integrated Gradients Blended Heat Map")
    viz.visualize_image_attr(attr_ig, None, method="heat_map",sign="all", show_colorbar=True, title="Integrated Gradients Heat Map")