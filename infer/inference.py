import os
import time
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# import the model (paste your ResUNet code here or import it)
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

infer_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class ResUNet(nn.Module):
    def __init__(self, n_classes):
        super(ResUNet, self).__init__()
        base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*base_layers[:3])  
        self.layer1 = nn.Sequential(*base_layers[3:5]) 
        self.layer2 = base_layers[5]                    
        self.layer3 = base_layers[6]                   
        self.layer4 = base_layers[7]                   

        self.upsample4 = self._upsample(512, 256)    
        self.upsample3 = self._upsample(256, 128)     
        self.upsample2 = self._upsample(128, 64)      
        self.upsample1 = self._upsample(64, 64)       

        
        self.upsample0 = self._upsample(64, 64)        

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def _upsample(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.layer0(x)  
        x1 = self.layer1(x0) 
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3) 

        x = self.upsample4(x4) + x3  
        x = self.upsample3(x) + x2  
        x = self.upsample2(x) + x1  
        x = self.upsample1(x) + x0  
        x = self.upsample0(x)       

        return self.final_conv(x)   

# -----------------------------
# 1) Select device (CPU or GPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUNet(n_classes=1).to(device)
model.load_state_dict(torch.load("infer/best_model.pth", map_location=device))
model.eval()

print("✅ Model loaded on:", device)

# -------------------------
# POST-PROCESSING: REMOVE SMALL COMPONENTS
# -------------------------
def remove_small_components(mask, min_area_pixels=None, min_area_ratio=0.001):
    """
    Remove small connected components from binary mask.
    
    Args:
        mask: Binary mask (uint8, 0 or 255)
        min_area_pixels: Minimum area in pixels (if None, uses min_area_ratio)
        min_area_ratio: Minimum area as ratio of total image pixels (default: 0.001 = 0.1%)
    
    Returns:
        Filtered binary mask
    """
    if min_area_pixels is None:
        total_pixels = mask.shape[0] * mask.shape[1]
        min_area_pixels = int(total_pixels * min_area_ratio)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create output mask
    filtered_mask = np.zeros_like(mask)
    
    # Keep only components larger than min_area
    for label_id in range(1, num_labels):  # Skip background (label 0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area_pixels:
            filtered_mask[labels == label_id] = 255
    
    return filtered_mask


# -------------------------
# INFERENCE FUNCTION
# -------------------------
def infer_single_image(model, img_path, min_area_ratio=0.001):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W = img_rgb.shape[:2]

    transformed = infer_transform(image=img_rgb)
    tensor_img = transformed["image"].unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        logits = model(tensor_img)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # resize prediction back to original image size
    prob = cv2.resize(prob, (W, H))

    # binary mask
    mask_bin = (prob > 0.5).astype(np.uint8) * 255
    
    # post-processing: remove small components
    mask_bin = remove_small_components(mask_bin, min_area_ratio=min_area_ratio)
    
    # update probability map to match filtered mask (optional, for visualization)
    # This ensures prob values are set to 0 where small components were removed
    prob_filtered = prob.copy()
    prob_filtered[mask_bin == 0] = 0

    return img, prob_filtered, mask_bin


# -------------------------
# OVERLAY MASK ON IMAGE
# -------------------------
def overlay_mask(image, prob, threshold=0.5):
    H, W = image.shape[:2]

    # binary mask
    mask_bin = (prob > threshold).astype(np.uint8)

    # create color mask (BLUE instead of green)
    mask_color = np.zeros_like(image)
    mask_color[:, :, 0] = mask_bin * 255  # Blue channel

    # overlay
    overlay = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)

    # calculate probability values
    max_prob = float(prob.max())

    # write probability text on image
    text = f"Max Prob: {max_prob:.3f}"

    # -------- AUTO SCALE TEXT --------
    # size of text relative to image height
    font_scale = max(0.3, H / 300)   # smaller images → smaller text
    thickness = max(1, int(H / 200)) # adaptive thickness
    y_offset = int(40 * font_scale)  # move down based on size

    # make sure text fits horizontally
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    if text_size[0] > W - 20:        # if too wide → shrink
        ratio = (W - 20) / text_size[0]
        font_scale *= ratio

    # draw text
    cv2.putText(
        overlay,
        text,
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return overlay


# -------------------------
# PROCESS DIRECTORY
# -------------------------
# Post-processing parameter: minimum area ratio (0.001 = 0.1% of image pixels)
# Adjust this value: smaller = more aggressive filtering, larger = less filtering
MIN_AREA_RATIO = 0.001  # You can change this value (e.g., 0.0005 for more filtering, 0.002 for less)

img_dir = "infer/img"
res_dir = "infer/res"
os.makedirs(res_dir, exist_ok=True)

image_files = [f for f in os.listdir(img_dir)
               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

total_time = 0
count = 0

print(f"Found {len(image_files)} images.")
print(f"Post-processing: removing components < {MIN_AREA_RATIO*100:.2f}% of image area")

for fname in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(img_dir, fname)

    start = time.time()
    img, prob, mask_bin = infer_single_image(model, img_path, min_area_ratio=MIN_AREA_RATIO)
    elapsed = time.time() - start

    total_time += elapsed
    count += 1

    overlay = overlay_mask(img, prob)

    save_path = os.path.join(res_dir, f"pred_{fname}")
    cv2.imwrite(save_path, overlay)


# -------------------------
# PRINT AVG TIME
# -------------------------
if count > 0:
    print("\n✅ Average inference time:", total_time / count, "seconds/image")