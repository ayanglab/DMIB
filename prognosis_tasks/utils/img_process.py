# Boilerplate imports.
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import torch
from torchvision import models, transforms
# From our repository.
import saliency.core as saliency

# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title, fontsize=10)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title, fontsize=10)

def ShowColorImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im,vmin=0, vmax=1)
    P.title(title, fontsize=10)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

def LoadImage_gray2color(file_path):
    im = PIL.Image.open(file_path).convert('RGB') # RGB； opencv： BGR
    # im = im.resize((299, 299))
    im = np.asarray(im)
    return im

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images)
    return images.requires_grad_(True)


def PreprocessImages_2D(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images =  np.asarray(images)
    images = np.expand_dims(images, axis=0)
    images = np.expand_dims(images, axis=0)
    images = images/255
    # images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    # return images.requires_grad_(True)
    return images

def PreprocessImages_test(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images =  np.asarray(images)
    images = np.expand_dims(images, axis=0)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    # return images.requires_grad_(True)
    return images


def PreprocessImages_adjust(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    # images = images.reshape(700,700,1)
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    return images.requires_grad_(True)

def PreprocessImages_cam(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    # images = images.reshape(350,350,1)
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    return images.requires_grad_(True)


# def conv_layer_forward(m, i, o):
#     # move the RGB dimension to the last dimension
#     conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().numpy()
#
#
# def conv_layer_backward(m, i, o):
#     # move the RGB dimension to the last dimension
#     conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().numpy()
conv_layer_outputs = {}
def conv_layer_forward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).cpu().detach().numpy()

def conv_layer_backward(m, i, o):
    # move the RGB dimension to the last dimension
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).cpu().detach().numpy()


class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = PreprocessImages(images)
    target_class_idx = call_model_args[class_idx_str]
    output = model(images)
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs

# def call_model_function_fyy(images, call_model_args=None, expected_keys=None):
#
#     images = PreprocessImages_adjust(images)
#     target_class_idx = call_model_args[class_idx_str]
#     output = model(images)
#     m = torch.nn.Softmax(dim=1)
#     output = m(output)
#     if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
#         outputs = output[:, target_class_idx]
#         grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
#         grads = torch.movedim(grads[0], 1, 3)
#         gradients = grads.detach().numpy()
#         return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
#     else:
#         one_hot = torch.zeros_like(output)
#         one_hot[:, target_class_idx] = 1
#         model.zero_grad()
#         output.backward(gradient=one_hot, retain_graph=True)
#         return conv_layer_outputs

if __name__ == '__main__':

    model = models.inception_v3(pretrained=True, init_weights=False)
    eval_mode = model.eval()

    # conv_layer = model.Mixed_7c
    # conv_layer_outputs = {}
    # conv_layer.register_forward_hook(conv_layer_forward)
    # conv_layer.register_full_backward_hook(conv_layer_backward)

    # Load the image
    im_orig = LoadImage('./doberman.png')
    im_tensor = PreprocessImages([im_orig])
    # Show the image
    ShowImage(im_orig)

    predictions = model(im_tensor)
    predictions = predictions.detach().numpy()
    prediction_class = np.argmax(predictions[0])
    call_model_args = {class_idx_str: prediction_class}

    print("Prediction class: " + str(prediction_class))  # Should be a doberman, class idx = 236
    im = im_orig.astype(np.float32)

    '''
    Vanilla Gradient & SmoothGrad
    '''
    # Construct the saliency object. This alone doesn't do anthing.
    gradient_saliency = saliency.GradientSaliency()

    # Compute the vanilla mask and the smoothed mask.
    vanilla_mask_3d = gradient_saliency.GetMask(im, call_model_function, call_model_args)
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, call_model_function, call_model_args)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 2
    UPSCALE_FACTOR = 10
    P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

    # Render the saliency masks.
    ShowGrayscaleImage(vanilla_mask_grayscale, title='Vanilla Gradient', ax=P.subplot(ROWS, COLS, 1))
    ShowGrayscaleImage(smoothgrad_mask_grayscale, title='SmoothGrad', ax=P.subplot(ROWS, COLS, 2))

