import numpy as np
import torch
from captum.attr import IntegratedGradients, DeepLift, Occlusion
from captum.attr import visualization as viz


methods = ["original_image", "blended_heat_map", "heat_map"]

# returns img and label at given index from the first batch of
# given validation dataloader
def getData(val_dl, index):
    images, labels = next(iter(val_dl))
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    imgTensor = images[index]
    imgTensor.requires_grad = True

    return imgTensor, labels[index]

# turns the given tensor input a plottable image by detaching
# transposing
def tensor2img(tensor):
    return np.transpose(tensor.cpu().detach().numpy(), (1, 2, 0))

# plots integrated gradients saliency map of given image
def integratedGrads(net, imgTensor, label):
    ig = IntegratedGradients(net)
    attr_ig = ig.attribute(imgTensor.unsqueeze(0), target=label)
    attr_ig = tensor2img(attr_ig.squeeze())

    signs = ["","all","all"]
    titles = ["Original Image", "Integrated Gradients - Blended Heat Map", "Integrated Gradients - Heat Map"]
    viz.visualize_image_attr_multiple(attr_ig, tensor2img(imgTensor), methods=methods, signs=signs, titles=titles,
                                      fig_size=(13,5), show_colorbar=True)

# plots deep lift saliency map of given image
def deepLift(net, imgTensor, label):
    dl = DeepLift(net)
    attr_dl = dl.attribute(imgTensor.unsqueeze(0), target=label)
    attr_dl = tensor2img(attr_dl.squeeze())

    signs = ["","all","all"]
    titles = ["Original Image", "DeepLift - Blended Heat Map", "DeepLift - Heat Map"]
    viz.visualize_image_attr_multiple(attr_dl, tensor2img(imgTensor), methods=methods, signs=signs, titles=titles,
                                      fig_size=(13,5), show_colorbar=True)

# plots occlusion saliency map of given image
def occlusionMap(net, imgTensor, label):
    occlusion = Occlusion(net)

    attr_occ = occlusion.attribute(imgTensor.unsqueeze(0), target=label, sliding_window_shapes=(3, 1, 1))
    attr_occ = tensor2img(attr_occ.squeeze())

    signs = ["", "positive", "positive"]
    titles = ["Original Image", "Occlusion - Blended Heat Map", "Occlusion - Heat Map"]
    viz.visualize_image_attr_multiple(attr_occ, tensor2img(imgTensor), methods=methods, signs=signs, titles=titles,
                                      fig_size=(13, 5), show_colorbar=True)
