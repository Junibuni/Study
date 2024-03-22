import torch
import torch.nn.functional as F
import numpy as np

def unnormalize(tensor, mean, std):
        copy_tensor = tensor.clone()
        for i in range(copy_tensor.shape[1]):
            copy_tensor[:, i] = copy_tensor[:, i] * std[i] + mean[i]

        return copy_tensor

def depth_gradient(depth_map):
    depth_tensor = depth_map.clone().detach()

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32)

    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

    device = depth_tensor.device
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)

    grad_x = F.conv2d(depth_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(depth_tensor, sobel_y, padding=1)

    gradient_vectors = torch.cat((grad_x, grad_y), dim=1)

    return gradient_vectors

def depth_gradient_loss(target_depth, output_depth):
    target_gradients = depth_gradient(target_depth)
    output_gradients = depth_gradient(output_depth)

    loss = F.l1_loss(target_gradients, output_gradients)

    return loss

if __name__ == "__main__":
    target_depth_map = torch.tensor([[[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]]], dtype=torch.float32)
    output_depth_map = torch.tensor([[[[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9.01]]]], dtype=torch.float32)
    
    loss = depth_gradient_loss(target_depth_map, output_depth_map)
    print("L1 Depth Gradient Loss:", loss.item())


