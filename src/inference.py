
# inference script for VOC segmentation

import os
import cv2
import torch
import numpy as np

from model import UNet


# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model path
model_path = "outputs/checkpoints/best_model.pth"


# test images folder
test_folder = "test_images"


# output folder
output_folder = "group27_output"

os.makedirs(output_folder, exist_ok=True)


# load model
model = UNet(num_classes=21)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


# preprocessing function
def preprocess(image):

    image = cv2.resize(image, (300,300))

    image = image / 255.0

    image = np.transpose(image, (2,0,1))

    image = torch.tensor(image).float()

    image = image.unsqueeze(0)

    return image


# inference loop
for img_name in os.listdir(test_folder):

    img_path = os.path.join(test_folder, img_name)

    image = cv2.imread(img_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_tensor = preprocess(image)

    input_tensor = input_tensor.to(device)


    with torch.no_grad():

        output = model(input_tensor)

        pred_mask = torch.argmax(output, dim=1)

        pred_mask = pred_mask.squeeze().cpu().numpy()


    # save mask
    mask_name = img_name.replace(".jpg","_mask.png")

    save_path = os.path.join(output_folder, mask_name)

    cv2.imwrite(save_path, pred_mask)


print("Inference completed.")