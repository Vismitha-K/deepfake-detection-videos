import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer=None, use_cuda=False):
        self.model = model
        self.model.eval()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()
        self._activations = None
        self._gradients = None

        if target_layer is None:
            target_layer = self._find_last_conv()
        self.target_layer = target_layer

        # forward / backward hooks
        def forward_hook(module, inp, outp):
            self._activations = outp.detach()
        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple
            self._gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # register backward hook with fallback
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(lambda module, grad_in, grad_out: backward_hook(module, grad_in, grad_out))
        else:
            self.target_layer.register_backward_hook(backward_hook)

    def _find_last_conv(self):
        for m in reversed(list(self.model.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No Conv2d layer found in model")

    def forward(self, input_tensor):
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        return self.model(input_tensor)

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: (1,C,H,W)
        """
        # forward + choose target
        output = self.forward(input_tensor)  # may be logits
        if target_class is None:
            # predicted class
            if output.dim() == 1 or output.size(1) == 1:
                target_class = 0
            else:
                target_class = int(output.softmax(dim=1)[0].argmax().cpu().numpy())

        one_hot = torch.zeros_like(output)
        if output.dim() == 1:
            one_hot[0] = 1.0
        else:
            one_hot[0, target_class] = 1.0

        # zero grads and backward to get gradients for target class
        self.model.zero_grad()
        # enable grad for backward
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self._gradients.cpu()[0]    # C x h x w
        acts  = self._activations.cpu()[0]  # C x h x w

        weights = grads.view(grads.size(0), -1).mean(dim=1)  # C
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)  # h x w
        for w, a in zip(weights, acts):
            cam += w * a
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        return cam.numpy()

def overlay_cam_on_image(img_bgr, cam, colormap=cv2.COLORMAP_JET, alpha=0.5):
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize((cam * 255).astype('uint8'), (w, h))
    heatmap = cv2.applyColorMap(cam_resized, colormap)
    overlay = cv2.addWeighted(heatmap, alpha, img_bgr, 1.0 - alpha, 0)
    return overlay