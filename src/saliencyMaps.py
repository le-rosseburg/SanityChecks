import numpy as np
import torch
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, DeepLift
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

# plots gradients saliency map of given image
def grads(net, imgTensor, label):
    saliency = Saliency(net)
    grads = saliency.attribute(imgTensor.unsqueeze(0), target=label)
    grads = tensor2img(grads.squeeze())

    signs = ["","absolute_value","absolute_value"]
    titles = ["Original Image", "Gradients - Blended Heat Map", "Gradients - Heat Map"]
    viz.visualize_image_attr_multiple(grads, tensor2img(imgTensor), methods=methods, signs=signs, titles=titles,
                                      fig_size=(13,5), show_colorbar=True)

# plots integrated gradients saliency map of given image
def integratedGrads(net, imgTensor, label):
    ig = IntegratedGradients(net)
    attr_ig = ig.attribute(imgTensor.unsqueeze(0), target=label)
    attr_ig = tensor2img(attr_ig.squeeze())

    signs = ["","all","all"]
    titles = ["Original Image", "Integrated Gradients - Blended Heat Map", "Integrated Gradients - Heat Map"]
    viz.visualize_image_attr_multiple(attr_ig, tensor2img(imgTensor), methods=methods, signs=signs, titles=titles,
                                      fig_size=(13,5), show_colorbar=True)

# plots integrated gradients with smooth gradient saliency map of given image
# not working on cuda, because of out of memory error
def integratedGradsSmoothGrad(net, imgTensor, label, disableCuda=True):
    if disableCuda:
        net.cpu()
        imgTensor = imgTensor.cpu()

    ig = IntegratedGradients(net)
    nt = NoiseTunnel(ig)
    attr_ig_nt = nt.attribute(imgTensor.unsqueeze(0), target=label, nt_type='smoothgrad_sq', nt_samples=100, stdevs=0.2)
    attr_ig_nt = tensor2img(attr_ig_nt.squeeze())

    signs = ["", "absolute_value", "absolute_value"]
    titles = ["Original Image", "Integrated Gradients wSG - Blended Heat Map", "Integrated Gradients wSG - Heat Map"]
    viz.visualize_image_attr_multiple(attr_ig_nt, tensor2img(imgTensor), methods=methods, signs=signs, titles=titles,
                                      fig_size=(13, 5), show_colorbar=True)

    if torch.cuda.is_available() and disableCuda: net.cuda()

# plots deep lift saliency map of given image
def deepLift(net, imgTensor, label):
    dl = DeepLift(net)
    attr_dl = dl.attribute(imgTensor.unsqueeze(0), target=label)
    attr_dl = tensor2img(attr_dl.squeeze())

    signs = ["","all","all"]
    titles = ["Original Image", "DeepLift - Blended Heat Map", "DeepLift - Heat Map"]
    viz.visualize_image_attr_multiple(attr_dl, tensor2img(imgTensor), methods=methods, signs=signs, titles=titles,
                                      fig_size=(13,5), show_colorbar=True)