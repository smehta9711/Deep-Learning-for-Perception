# import os
# import shutil
import numpy as np
# from PIL import Image
import cv2


def dp(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)
            
def CombineImages(pred, label, rgb):
    pred = pred.detach().cpu().numpy().squeeze()
    label = label.detach().cpu().numpy().squeeze()
    rgb = rgb.detach().cpu().numpy()
    
    gray_array = 0.299 * rgb[0, :, :] + 0.587 * rgb[1, :, :] + 0.114 * rgb[2, :, :]

    # target_shape = gray_array.shape

    # pred_resized = cv2.resize(pred, (target_shape[1], target_shape[0]))
    # label_resized = cv2.resize(label, (target_shape[1], target_shape[0]))
    
    # Concatenate images horizontally
    # combined_image_np = np.concatenate((pred, label, gray_array), axis=1)
    # combined_image_np = (np.clip(combined_image_np, 0, 1)*255).astype(np.uint8)

    # combined_image_np = np.concatenate((pred_resized, label_resized, gray_array), axis=1)
    combined_image_np = np.concatenate((pred, label, gray_array), axis=1)

    # Clip values between 0 and 1, then scale to 255 and convert to uint8
    combined_image_np = (np.clip(combined_image_np, 0, 1) * 255).astype(np.uint8)
    return combined_image_np