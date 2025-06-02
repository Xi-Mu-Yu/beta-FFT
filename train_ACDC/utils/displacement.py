
import torch
import torch.fft

def get_low_freq_component_single(image_tensor, threshold):
    """
    获取单张图像的低频区域并转换回空间域
    :param image_tensor: 输入的图像张量，形状为 (1, H, W)
    :param threshold: 阈值，用于生成低频掩码
    :return: 低频区域的空间域表示，形状为 (1, H, W)
    """
    # 获取图像的形状
    _,_, rows, cols = image_tensor.shape
    
    # 转换到频域
    f_transform = torch.fft.fft2(image_tensor, dim=(-2, -1))
    f_shift = torch.fft.fftshift(f_transform, dim=(-2, -1))
    
    # 生成低频掩码
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros((rows, cols), dtype=torch.float32, device=image_tensor.device)
    mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1
    
    # 应用掩码并进行逆傅里叶变换，得到低频部分的图像
    f_low_freq = f_shift * mask
    f_ishift = torch.fft.ifftshift(f_low_freq, dim=(-2, -1))
    img_back = torch.abs(torch.fft.ifft2(f_ishift, dim=(-2, -1)))
    
    return img_back.unsqueeze(0)

def process_image_batches(image_batch_with_labels_tensor, image_batch_without_labels_tensor, threshold):
    """
    处理多个图像批次，提取并交换每个批次第 i 张图像的低频区域
    :param image_batch_with_labels_tensor: 带标签的图像张量（批量），形状为 (N, 1, H, W)
    :param image_batch_without_labels_tensor: 不带标签的图像张量（批量），形状为 (N, 1, H, W)
    :param threshold: 阈值，决定低频区域的大小
    :return: 互换低频区域后的两个图像张量（批量）
    """
    # 确保两个批次具有相同的大小
    assert image_batch_with_labels_tensor.shape[0] == image_batch_without_labels_tensor.shape[0], \
        "两个批次的大小必须相同"
    
    batch_size = image_batch_with_labels_tensor.shape[0]
    
    new_batch_with_labels = []
    new_batch_without_labels = []
    
    # 对每张图像进行处理
    for i in range(batch_size):
        # 提取第 i 张图像
        img_with_labels = image_batch_with_labels_tensor[i:i+1]
        img_without_labels = image_batch_without_labels_tensor[i:i+1]
        
        # print("img_with_labels大小",img_with_labels.shape)
        
        # 计算低频区域
        low_freq_with_labels = get_low_freq_component_single(img_with_labels, threshold)
        low_freq_without_labels = get_low_freq_component_single(img_without_labels, threshold)
        
        # 交换低频区域
        swapped_img_with_labels = img_with_labels - low_freq_with_labels + low_freq_without_labels
        swapped_img_without_labels = img_without_labels - low_freq_without_labels + low_freq_with_labels
        
        # 将处理后的图像加入新的批次中
        new_batch_with_labels.append(swapped_img_with_labels)
        new_batch_without_labels.append(swapped_img_without_labels)
    
    # 将列表转换为张量
    new_batch_with_labels = torch.cat(new_batch_with_labels, dim=0)
    new_batch_without_labels = torch.cat(new_batch_without_labels, dim=0)
    last_image = torch.cat([new_batch_with_labels, new_batch_without_labels], dim=0)
    return last_image.view(-1,256,256)

