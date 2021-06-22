import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz


def tensor2img(tensor):
    return np.transpose(tensor.cpu().detach().numpy(), (1, 2, 0))

def main(net, val_dl, index):
    net.eval()

    images, labels = next(iter(val_dl))
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    """Plot saliency map:"""
    ig = IntegratedGradients(net)
    attr_ig = ig.attribute(images[index].unsqueeze(0), target=labels[index])
    attr_ig = tensor2img(attr_ig.squeeze())

    original_img = tensor2img(images[index])
    methods = ["original_image", "blended_heat_map", "heat_map"]
    signs = ["","all","all"]
    titles = ["Original Image", "Integrated Gradients - Blended Heat Map", "Integrated Gradients - Heat Map"]
    viz.visualize_image_attr_multiple(attr_ig, original_img, methods=methods, signs=signs, titles=titles,
                                      fig_size=(13,5), show_colorbar=True)