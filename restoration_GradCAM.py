ort
torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_image, criterion):
        self.model.eval()

        # Forward pass
        output = self.model(input_image)
        loss = criterion(output, target_image)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Compute gradients and weights
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        weights = np.mean(gradients, axis=(2, 3))[0, :]

        # Compute Grad-CAM
        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam


# 이미지 시각화 및 저장
def visualize_and_save_cam(input_image, cam, save_path):
    input_image = input_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    input_image = input_image - np.min(input_image)
    input_image = input_image / np.max(input_image)

    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) / 255
    cam = cam + np.float32(input_image)
    cam = cam / np.max(cam)

    plt.imshow(cam)
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()


# 모델, 입력 이미지, 타겟 이미지, 손실 함수 준비
model = SimpleUNet()
input_image = torch.randn(1, 1, 224, 224)  # 예제 입력 이미지
target_image = torch.randn(1, 1, 224, 224)  # 예제 타겟 이미지
criterion = torch.nn.MSELoss()

# Grad-CAM 생성 및 시각화
grad_cam = GradCAM(model, model.encoder[3])  # 마지막 Conv 레이어
cam = grad_cam.generate_cam(input_image, target_image, criterion)
visualize_and_save_cam(input_image, cam, 'grad_cam.png')