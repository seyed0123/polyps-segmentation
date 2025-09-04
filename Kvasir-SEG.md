```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
```

# EDA


```python
IMAGE_DIR1 = "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images"
MASK_DIR1 = "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks"

IMAGE_DIR2 = '/kaggle/input/cvcclinicdb/PNG/Original'
MASK_DIR2 = '/kaggle/input/cvcclinicdb/PNG/Ground Truth'

IMAGE_DIR3 = '/kaggle/input/merged-polyp-segmentation-datasets/images_valid/images_valid'
MASK_DIR3 =  '/kaggle/input/merged-polyp-segmentation-datasets/images_valid/masks_valid'

image_paths1 = sorted(glob(os.path.join(IMAGE_DIR1, "*.jpg")))
mask_paths1 = sorted(glob(os.path.join(MASK_DIR1, "*.jpg")))

print(f"Kvasir -> {len(image_paths1)} images, {len(mask_paths1)} masks")

image_paths2 = sorted(glob(os.path.join(IMAGE_DIR2, "*.png")))
mask_paths2  = sorted(glob(os.path.join(MASK_DIR2, "*.png")))

print(f"CVC -> {len(image_paths2)} images, {len(mask_paths2)} masks")

image_paths3 = sorted(glob(os.path.join(IMAGE_DIR3, "*.png")))
mask_paths3  = sorted(glob(os.path.join(MASK_DIR3, "*.png")))

print(f"MPSD -> {len(image_paths3)} images, {len(mask_paths3)} masks")

image_paths = image_paths1+image_paths2+image_paths3
mask_paths = mask_paths1 + mask_paths2 + mask_paths3
image_names = [os.path.basename(p) for p in image_paths]
mask_names = [os.path.basename(p) for p in mask_paths]
assert image_names == mask_names, "Image-mask filenames do not match!"
```

    Kvasir -> 1000 images, 1000 masks
    CVC -> 612 images, 612 masks
    MPSD -> 22330 images, 22330 masks
    


```python
data_info = []

def process_dataset(image_paths, mask_paths, dataset_name):
    for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc=f"Processing {dataset_name}"):
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        
        h, w = img_np.shape[:2]
        
        
        mask_bin = (mask_np > 0).astype(np.uint8)
        coverage_ratio = mask_bin.sum() / (h * w)
        
        data_info.append({
            "filename": os.path.basename(img_path),
            "image_path": img_path,
            "mask_path": mask_path,
            "dataset": dataset_name,
            "width": w,
            "height": h,
            "aspect_ratio": w / h,
            "mask_coverage": coverage_ratio
        })



process_dataset(image_paths1, mask_paths1, "kvasir")
process_dataset(image_paths2, mask_paths2, "cvc")
process_dataset(image_paths3, mask_paths3, "mpsd")

df = pd.DataFrame(data_info)
```

    Processing kvasir: 100%|██████████| 1000/1000 [00:21<00:00, 46.35it/s]
    Processing cvc: 100%|██████████| 612/612 [00:10<00:00, 59.63it/s]
    Processing mpsd: 100%|██████████| 22330/22330 [03:37<00:00, 102.84it/s]
    


```python
print("\n=== Dataset Summary ===")
print(df.describe())
```

    
    === Dataset Summary ===
                  width        height  aspect_ratio  mask_coverage
    count  23942.000000  23942.000000  23942.000000   23942.000000
    mean     274.696350    268.898338      1.014547       0.113924
    std       80.415834     60.804019      0.061155       0.119316
    min      256.000000    256.000000      0.681725       0.000748
    25%      256.000000    256.000000      1.000000       0.033173
    50%      256.000000    256.000000      1.000000       0.074475
    75%      256.000000    256.000000      1.000000       0.152172
    max     1920.000000   1072.000000      1.791045       1.469482
    


```python
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df["width"], bins=20, kde=False)
plt.title("Image Width Distribution")

plt.subplot(1,2,2)
sns.histplot(df["height"], bins=20, kde=False)
plt.title("Image Height Distribution")
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_6_0.png)
    



```python
plt.figure(figsize=(6,4))
sns.histplot(df["aspect_ratio"], bins=20, kde=True)
plt.title("Aspect Ratio Distribution")
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_7_0.png)
    



```python
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="width", y="height", alpha=0.6)
plt.title("Image Resolution Distribution")
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_8_0.png)
    



```python
plt.figure(figsize=(6,4))
sns.histplot(df["mask_coverage"], bins=30, kde=True)
plt.axvline(df["mask_coverage"].mean(), color='r', linestyle='--', label="Mean")
plt.legend()
plt.title("Distribution of Polyp Coverage")
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_9_0.png)
    



```python
plt.figure(figsize=(6,4))
sns.boxplot(x=df["mask_coverage"])
plt.title("Boxplot of Polyp Coverage Ratio")
plt.xlabel("Mask coverage ratio")
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_10_0.png)
    



```python
empty_count = (df["mask_coverage"] <= 0.005).sum()
non_empty_count = (df["mask_coverage"] > 0.005).sum()

print(f"Empty masks: {empty_count} ({empty_count/len(df)*100:.2f}%)")
print(f"Non-empty masks: {non_empty_count} ({non_empty_count/len(df)*100:.2f}%)")
```

    Empty masks: 460 (1.92%)
    Non-empty masks: 23482 (98.08%)
    


```python
sample_imgs = np.random.choice(image_paths, size=100, replace=False)
pixels = []
for p in sample_imgs:
    img = cv2.imread(p)  # BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels.append(img_rgb.reshape(-1, 3))
pixels = np.vstack(pixels)

plt.figure(figsize=(8,5))
plt.hist(pixels[:,0], bins=50, color='r', alpha=0.5, label='Red')
plt.hist(pixels[:,1], bins=50, color='g', alpha=0.5, label='Green')
plt.hist(pixels[:,2], bins=50, color='b', alpha=0.5, label='Blue')
plt.title("Pixel Intensity Distribution (RGB channels)")
plt.legend()
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_12_0.png)
    



```python
means, stds = [], []
for img_path in np.random.choice(image_paths, 200, replace=False):
    img = np.array(Image.open(img_path)) / 255.0
    means.append(img.mean())
    stds.append(img.std())

plt.figure(figsize=(6,4))
sns.scatterplot(x=means, y=stds)
plt.xlabel("Mean Intensity")
plt.ylabel("Std Dev Intensity")
plt.title("Image Brightness vs Contrast")
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_13_0.png)
    



```python
mask_values = []
for mask_path in np.random.choice(mask_paths, size=50, replace=False):
    mask = np.array(Image.open(mask_path))
    unique_vals = np.unique(mask)
    mask_values.extend(unique_vals.tolist())

mask_values = np.unique(mask_values)
print("\nUnique pixel values in masks:", mask_values)
if np.array_equal(mask_values, [0, 255]) or np.array_equal(mask_values, [0,1]):
    print("✅ Masks are binary.")
else:
    print("⚠️ Masks have unexpected values!")
```

    
    Unique pixel values in masks: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  18
      19  21  23  24  25  26  28  29  30  31  32  38  44  49  50  51  53  56
      64  66  73  77  80  82  85  88  91  96  98 112 248 249 250 251 252 253
     254 255]
    ⚠️ Masks have unexpected values!
    


```python
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="width", y="mask_coverage", alpha=0.7)
plt.title("Mask Coverage vs Image Width")
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_15_0.png)
    



```python
def visualize_samples(num_samples=5):
    samples = np.random.choice(len(image_paths), size=num_samples, replace=False)
    plt.figure(figsize=(15, num_samples*3))
    for i, idx in enumerate(samples):
        img = Image.open(image_paths[idx])
        mask = Image.open(mask_paths[idx])
        
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image: {os.path.basename(image_paths[idx])}")
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(img)
        plt.imshow(mask, cmap='Reds', alpha=0.5)
        plt.axis("off")
        plt.title("Overlay Mask")
    plt.tight_layout()
    plt.show()

visualize_samples(5)
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_16_0.png)
    



```python
def load_image_and_mask(row):
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    return img, mask

bins = pd.qcut(df["mask_coverage"], q=4, labels=["small", "medium", "large", 'huge'])
sample_idx = df.groupby(bins).apply(lambda x: x.sample(3, random_state=1)).index.get_level_values(1)

fig, axes = plt.subplots(4, 3, figsize=(12, 12))
for i, idx in enumerate(sample_idx):
    img, mask = load_image_and_mask(df.iloc[idx])
    ax = axes[i // 3, i % 3]
    ax.imshow(img)
    ax.imshow(mask, cmap="Reds", alpha=0.4)
    ax.set_title(f"{bins.iloc[idx]} ({df['mask_coverage'].iloc[idx]:.2%})")
    ax.axis("off")
plt.tight_layout()
```

    /tmp/ipykernel_19/2861692189.py:9: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      sample_idx = df.groupby(bins).apply(lambda x: x.sample(3, random_state=1)).index.get_level_values(1)
    


    
![png](Kvasir-SEG_files/Kvasir-SEG_17_1.png)
    



```python
row = df[df['dataset']=='kvasir'].sample(1, random_state=0).iloc[0]
img = np.array(Image.open(row["image_path"]).convert("RGB"))
mask = np.array(Image.open(row["mask_path"]).convert("L"))


ys, xs = np.where(mask > 0)
if len(xs) > 0 and len(ys) > 0:

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    

    pad = 10
    x_min, x_max = max(0, x_min-pad), min(img.shape[1], x_max+pad)
    y_min, y_max = max(0, y_min-pad), min(img.shape[0], y_max+pad)
    
    masked_crop = img[y_min:y_max, x_min:x_max]
else:
 
    masked_crop = img

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(masked_crop)
ax1.set_title("Zoomed Polyp Region")
ax1.axis("off")

ax2.imshow(img)
ax2.imshow(mask, cmap="Reds", alpha=0.4)
ax2.set_title("Full Image with Mask Overlay")
ax2.axis("off")

plt.tight_layout()
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_18_0.png)
    



```python
grid_size = 5
fig, axes = plt.subplots(grid_size, grid_size, figsize=(10,10))
imgs = df.sample(n=grid_size**2, random_state=0)
for ax, (_, row) in zip(axes.flatten(), imgs.iterrows()):
    img = np.array(Image.open(row['image_path'])).astype(float)/255
    mean, std = img.mean(), img.std()
    ax.imshow(img)
    ax.set_title(f"μ={mean:.2f}, σ={std:.2f}", fontsize=8)
    ax.axis('off')
plt.suptitle("Intensity Fingerprints")
plt.tight_layout()
plt.show()
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_19_0.png)
    


## hypo testing

### Kolmogorov–Smirnov Test: Are polyp sizes uniformly distributed?

#### Purpose

We want to understand whether polyp sizes in the dataset follow a normal distribution. This is important because many statistical methods assume normality, and model strategies (like data augmentation or stratified sampling) may depend on the distribution of polyp sizes. In medical imaging, datasets often have a skew toward small lesions, which can affect model performance.

#### Hypotheses

H₀ (null): Polyp coverage follows a normal distribution.

H₁ (alternative): Polyp coverage does not follow a normal distribution.


```python
from scipy.stats import shapiro
stat, p = shapiro(df["mask_coverage"])

print(f"KS test (coverage vs uniform): {p:.3e}")
```

    KS test (coverage vs uniform): 5.662e-100
    

Polyp coverage does not follow a normal distribution. The data is strongly skewed, with most polyps being small and a few large ones.

### Spearman Correlation: Do larger images tend to have larger polyps?

#### Purpose:
Since datasets often contain images from different cameras or settings, larger image resolutions might be associated with larger polyps. Understanding this helps us know if resolution introduces bias in polyp size distribution.

#### Hypotheses:

H₀: There is no monotonic relationship between image resolution (width × height) and polyp size (coverage).

H₁: There is a monotonic relationship between resolution and polyp size.


```python
from scipy.stats import spearmanr

corr, p = spearmanr(df["width"]*df["height"], df["mask_coverage"])
print(f"Spearman correlation (Resolution vs Coverage): {corr:.3f}, p={p:.3e}")
```

    Spearman correlation (Resolution vs Coverage): 0.169, p=1.071e-153
    

There is a weak but significant positive correlation: larger-resolution images tend to have slightly larger polyps.
This is not a strong biological effect but may indicate that some datasets (with higher resolution) contain larger lesions. For model building, this could create dataset bias where resolution and lesion size are confounded.

### T-test: Do portrait vs. landscape images differ in polyp coverage?

#### Purpose:
Images come in both portrait and landscape orientations. If orientation influences polyp coverage, it may point to differences in how images were captured or cropped (scanner/device bias).

#### Hypotheses:

H₀: Mean polyp coverage is the same in portrait and landscape images.

H₁: Mean polyp coverage is different between portrait and landscape images.


```python
portrait = df[df["aspect_ratio"] < 1]["mask_coverage"]
landscape = df[df["aspect_ratio"] >= 1]["mask_coverage"]

from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(portrait, landscape, equal_var=False)
print(f"T-test (portrait vs landscape) p={p_val:.3f}")
```

    T-test (portrait vs landscape) p=0.028
    

There is a statistically significant difference in polyp coverage between portrait and landscape images. The difference is small but suggests that orientation may introduce bias. For training, mixing orientations without balancing could lead to subtle model biases.

### Pearson Correlation: Is polyp size related to image contrast?

#### Purpose:
Image contrast (measured by pixel intensity standard deviation) affects model learning. We want to know whether larger polyps also tend to have higher/lower contrast, which could affect segmentation difficulty.

#### Hypotheses:

H₀: There is no linear relationship between polyp size and image contrast.

H₁: There is a linear relationship between polyp size and image contrast.


```python
from scipy.stats import pearsonr
stds = []
for img_path in image_paths:
    img = np.array(Image.open(img_path)) / 255.0
    stds.append(img.std())
df["intensity_std"] = stds

corr, p = pearsonr(df["mask_coverage"], df["intensity_std"])
print(f"Correlation (mask coverage vs contrast): {corr:.3f}, p={p:.3e}")
```

    Correlation (mask coverage vs contrast): 0.081, p=2.042e-36
    

There is a very weak but statistically significant positive correlation: larger polyps tend to have slightly higher contrast. This could mean that bigger lesions have more visible texture variation, but the effect is small. From a modeling perspective, this correlation is unlikely to provide strong predictive power.

# Preprosessing


```python
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
```


```python
class PolypDataset(Dataset):
    def __init__(self, datasets, transform=None, exts=("png","jpg","jpeg")):
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        for image_dir, mask_dir in datasets:
            image_paths, mask_paths = self._load_dataset(image_dir, mask_dir, exts)
            self.image_paths.extend(image_paths)
            self.mask_paths.extend(mask_paths)

    def _load_dataset(self, image_dir, mask_dir, exts):
        
        image_paths = []
        for ext in exts:
            image_paths.extend(glob(os.path.join(image_dir, f"*.{ext}")))
        image_paths = sorted(image_paths)

        
        mask_paths = []
        for ext in exts:
            mask_paths.extend(glob(os.path.join(mask_dir, f"*.{ext}")))
        mask_paths = sorted(mask_paths)

        
        image_names = [os.path.basename(p) for p in image_paths]
        mask_dict = {os.path.basename(p): p for p in mask_paths}
        matched = [(img, mask_dict[os.path.basename(img)]) for img in image_paths if os.path.basename(img) in mask_dict]

        if matched:
            imgs, masks = zip(*matched)
        else:
            imgs, masks = [], []

        return list(imgs), list(masks)

    def __len__(self):
        return len(self.image_paths)

    def pad_to_square(self, img, color=(0, 0, 0)):
        h, w = img.shape[:2]
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    def __getitem__(self, idx):
        
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        
        image = self.pad_to_square(image)
        mask = self.pad_to_square(mask, color=(0,))

        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.long()
```


```python
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.2),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=15, 
        p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
    ),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

    /tmp/ipykernel_19/3157186758.py:5: UserWarning: Argument(s) 'value, mask_value' are not valid for transform ShiftScaleRotate
      A.ShiftScaleRotate(
    


```python
from torch.utils.data import random_split

datasets = [
    ("/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images",
     "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks"),

    ("/kaggle/input/cvcclinicdb/PNG/Original",
     "/kaggle/input/cvcclinicdb/PNG/Ground Truth"),

    ("/kaggle/input/merged-polyp-segmentation-datasets/images_valid/images_valid",
     "/kaggle/input/merged-polyp-segmentation-datasets/images_valid/masks_valid"),
]

full_dataset = PolypDataset(datasets, transform=None)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


train_dataset.dataset.transform = train_transform  
val_dataset.dataset.transform = val_transform     

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)


for images, masks in train_loader:
    print(images.shape)  # (B, 3, 512, 512)
    print(masks.shape)   # (B, 512, 512)
    break
```

    torch.Size([8, 3, 256, 256])
    torch.Size([8, 256, 256])
    

# train baseline model


```python
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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

```


```python
def dice_loss_from_logits_1c(logits, target, eps=1e-6):
    
    prob = torch.sigmoid(logits)
    inter = (prob*target).sum(dim=(1,2,3))
    den   = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1 - ((2*inter + eps) / (den + eps)).mean()

def dice_coeff(pred, target):
    smooth = 1.
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred) if pred.dim() == 4 else pred  # apply sigmoid if logits
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum(dim=[1,2])   # sum over spatial dims
    union = pred.sum(dim=[1,2]) + target.sum(dim=[1,2]) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

```


```python
def plot_res(train_losses,val_losses,val_dice,val_iou):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # --- Loss ---
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Learning Curve - Loss")
    
    # --- Dice ---
    axes[1].plot(val_dice, label="Val Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].legend()
    axes[1].set_title("Learning Curve - Dice Score")
    
    # --- IOU ---
    axes[2].plot(val_iou, label="Val IOU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IOU metric")
    axes[2].legend()
    axes[2].set_title("Learning Curve - IOU Score")
    
    plt.tight_layout()
    plt.show()
```


```python
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUNet(n_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses, val_dice,val_iou = [], [], [],[]
best_val_dice = 0.0
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader):
        images = images.to(device)
        
        masks = (masks > 0).long().to(device)

        optimizer.zero_grad()
        outputs = model(images)  
        loss = F.cross_entropy(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss, dice,iou = 0, 0,0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = (masks > 0).long().to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, masks)
            val_loss += loss.item()
            pred_logits = outputs[:, 1, :, :]
            dice += dice_coeff(pred_logits, masks).item()
            iou += iou_score(pred_logits, masks)

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_dice.append(dice / len(val_loader))
    val_iou.append(iou/ len(val_loader))

    if dice > best_val_dice:
        best_val_dice = dice
        torch.save(model.state_dict(), "base_model.pth")
        print("✅ Saved best model!")
    

    print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Dice: {val_dice[-1]:.4f} | IOU: {val_iou[-1]:.4f}")

```

    Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
    100%|██████████| 83.3M/83.3M [00:01<00:00, 77.4MB/s]
    100%|██████████| 2395/2395 [02:24<00:00, 16.56it/s]
    

    ✅ Saved best model!
    Epoch 1 | Train Loss: 0.1186 | Val Loss: 0.0669 | Dice: 0.8716 | IOU: 0.7335
    

    100%|██████████| 2395/2395 [02:24<00:00, 16.53it/s]
    

    Epoch 2 | Train Loss: 0.0583 | Val Loss: 0.0796 | Dice: 0.8556 | IOU: 0.7601
    

    100%|██████████| 2395/2395 [02:25<00:00, 16.42it/s]
    

    ✅ Saved best model!
    Epoch 3 | Train Loss: 0.0444 | Val Loss: 0.0547 | Dice: 0.8893 | IOU: 0.7844
    

    100%|██████████| 2395/2395 [02:25<00:00, 16.48it/s]
    

    ✅ Saved best model!
    Epoch 4 | Train Loss: 0.0372 | Val Loss: 0.0446 | Dice: 0.9095 | IOU: 0.8128
    

    100%|██████████| 2395/2395 [02:24<00:00, 16.53it/s]
    

    ✅ Saved best model!
    Epoch 5 | Train Loss: 0.0326 | Val Loss: 0.0453 | Dice: 0.9130 | IOU: 0.7988
    

    100%|██████████| 2395/2395 [02:24<00:00, 16.59it/s]
    

    ✅ Saved best model!
    Epoch 6 | Train Loss: 0.0297 | Val Loss: 0.0415 | Dice: 0.9209 | IOU: 0.8302
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.75it/s]
    

    Epoch 7 | Train Loss: 0.0268 | Val Loss: 0.0444 | Dice: 0.9153 | IOU: 0.8138
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.71it/s]
    

    ✅ Saved best model!
    Epoch 8 | Train Loss: 0.0259 | Val Loss: 0.0402 | Dice: 0.9313 | IOU: 0.8318
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.65it/s]
    

    ✅ Saved best model!
    Epoch 9 | Train Loss: 0.0240 | Val Loss: 0.0375 | Dice: 0.9324 | IOU: 0.8361
    

    100%|██████████| 2395/2395 [02:22<00:00, 16.79it/s]
    

    Epoch 10 | Train Loss: 0.0218 | Val Loss: 0.0441 | Dice: 0.9275 | IOU: 0.8189
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.70it/s]
    

    Epoch 11 | Train Loss: 0.0220 | Val Loss: 0.0448 | Dice: 0.9246 | IOU: 0.8078
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.73it/s]
    

    Epoch 12 | Train Loss: 0.0192 | Val Loss: 0.0474 | Dice: 0.9195 | IOU: 0.8193
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.66it/s]
    

    ✅ Saved best model!
    Epoch 13 | Train Loss: 0.0191 | Val Loss: 0.0370 | Dice: 0.9431 | IOU: 0.8552
    

    100%|██████████| 2395/2395 [02:22<00:00, 16.75it/s]
    

    Epoch 14 | Train Loss: 0.0165 | Val Loss: 0.0421 | Dice: 0.9269 | IOU: 0.8307
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.74it/s]
    

    Epoch 15 | Train Loss: 0.0168 | Val Loss: 0.0354 | Dice: 0.9419 | IOU: 0.8679
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.65it/s]
    

    Epoch 16 | Train Loss: 0.0164 | Val Loss: 0.0403 | Dice: 0.9400 | IOU: 0.8548
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.69it/s]
    

    Epoch 17 | Train Loss: 0.0170 | Val Loss: 0.0422 | Dice: 0.9382 | IOU: 0.8442
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.66it/s]
    

    Epoch 18 | Train Loss: 0.0155 | Val Loss: 0.0379 | Dice: 0.9424 | IOU: 0.8599
    

    100%|██████████| 2395/2395 [02:24<00:00, 16.62it/s]
    

    Epoch 19 | Train Loss: 0.0130 | Val Loss: 0.0409 | Dice: 0.9409 | IOU: 0.8629
    

    100%|██████████| 2395/2395 [02:23<00:00, 16.70it/s]
    

    Epoch 20 | Train Loss: 0.0129 | Val Loss: 0.0400 | Dice: 0.9411 | IOU: 0.8637
    


```python
plot_res(train_losses,val_losses,val_dice,val_iou)
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_47_0.png)
    



```python
def predict_tta(model, x):
    outs=[]
    with torch.no_grad():
        o = model(x)
        outs.append(o)
        o = model(torch.flip(x,[3]));        outs.append(torch.flip(o,[3]))
        o = model(torch.flip(x,[2]));        outs.append(torch.flip(o,[2]))
        o = model(x.transpose(2,3));         outs.append(model(x.transpose(2,3)).transpose(2,3))
    return torch.stack(outs).mean(0) 
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUNet(n_classes=1).to(device)
train_losses, val_losses, val_dice,val_iou = [], [], [],[]

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=True
)

pos_w = torch.tensor([2.0], device=device) 
best_val_dice = 0.0


for epoch in range(num_epochs):
    
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        images = images.to(device)                                 
        masks  = (masks > 0).float().unsqueeze(1).to(device)       

        optimizer.zero_grad()
        logits = model(images)                                    

        bce  = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=pos_w)
        loss = bce + 0.5 * dice_loss_from_logits_1c(logits, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

   
    model.eval()
    val_loss, dice, iou = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            images = images.to(device)
            masks  = (masks > 0).float().unsqueeze(1).to(device)  

            logits = predict_tta(model, images)

            bce  = F.binary_cross_entropy_with_logits(logits, masks, pos_weight=pos_w)
            loss = bce + 0.5 * dice_loss_from_logits_1c(logits, masks)

            val_loss += loss.item()
            dice += dice_coeff(logits, masks).item()
            iou  += iou_score(logits, masks)

    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    dice /= len(val_loader)
    iou  /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_dice.append(dice)
    val_iou.append(iou)

   
    scheduler.step(val_loss)

    
    if dice > best_val_dice:
        best_val_dice = dice
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Saved best model!")

    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
          f"Dice {dice:.4f} | IoU {iou:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")
```

    /usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
      warnings.warn(
    Epoch 1/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.45it/s]
    Epoch 1/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.15it/s]
    

    ✅ Saved best model!
    Epoch 1/20 | Train 0.3219 | Val 0.1693 | Dice 0.8998 | IoU 0.9055 | LR 1.00e-04
    

    Epoch 2/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.43it/s]
    Epoch 2/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.19it/s]
    

    ✅ Saved best model!
    Epoch 2/20 | Train 0.1615 | Val 0.1322 | Dice 0.9192 | IoU 0.9186 | LR 1.00e-04
    

    Epoch 3/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.47it/s]
    Epoch 3/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.21it/s]
    

    ✅ Saved best model!
    Epoch 3/20 | Train 0.1196 | Val 0.1135 | Dice 0.9327 | IoU 0.9297 | LR 1.00e-04
    

    Epoch 4/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.45it/s]
    Epoch 4/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.21it/s]
    

    ✅ Saved best model!
    Epoch 4/20 | Train 0.0984 | Val 0.0967 | Dice 0.9416 | IoU 0.9374 | LR 1.00e-04
    

    Epoch 5/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.50it/s]
    Epoch 5/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.20it/s]
    

    ✅ Saved best model!
    Epoch 5/20 | Train 0.0833 | Val 0.0964 | Dice 0.9439 | IoU 0.9401 | LR 1.00e-04
    

    Epoch 6/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.48it/s]
    Epoch 6/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.19it/s]
    

    ✅ Saved best model!
    Epoch 6/20 | Train 0.0727 | Val 0.0854 | Dice 0.9482 | IoU 0.9429 | LR 1.00e-04
    

    Epoch 7/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.43it/s]
    Epoch 7/20 [Val]: 100%|██████████| 599/599 [00:54<00:00, 11.09it/s]
    

    Epoch 7/20 | Train 0.0646 | Val 0.0916 | Dice 0.9477 | IoU 0.9446 | LR 1.00e-04
    

    Epoch 8/20 [Train]: 100%|██████████| 2395/2395 [02:26<00:00, 16.32it/s]
    Epoch 8/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.20it/s]
    

    ✅ Saved best model!
    Epoch 8/20 | Train 0.0586 | Val 0.0864 | Dice 0.9514 | IoU 0.9472 | LR 1.00e-04
    

    Epoch 9/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.51it/s]
    Epoch 9/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.20it/s]
    

    ✅ Saved best model!
    Epoch 9/20 | Train 0.0539 | Val 0.0802 | Dice 0.9552 | IoU 0.9500 | LR 1.00e-04
    

    Epoch 10/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.49it/s]
    Epoch 10/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.14it/s]
    

    Epoch 10/20 | Train 0.0486 | Val 0.0892 | Dice 0.9531 | IoU 0.9493 | LR 1.00e-04
    

    Epoch 11/20 [Train]: 100%|██████████| 2395/2395 [02:32<00:00, 15.75it/s]
    Epoch 11/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.11it/s]
    

    ✅ Saved best model!
    Epoch 11/20 | Train 0.0452 | Val 0.0755 | Dice 0.9572 | IoU 0.9518 | LR 1.00e-04
    

    Epoch 12/20 [Train]: 100%|██████████| 2395/2395 [02:27<00:00, 16.29it/s]
    Epoch 12/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.19it/s]
    

    Epoch 12/20 | Train 0.0435 | Val 0.0787 | Dice 0.9571 | IoU 0.9520 | LR 1.00e-04
    

    Epoch 13/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.42it/s]
    Epoch 13/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.18it/s]
    

    Epoch 13/20 | Train 0.0388 | Val 0.0814 | Dice 0.9565 | IoU 0.9524 | LR 1.00e-04
    

    Epoch 14/20 [Train]: 100%|██████████| 2395/2395 [02:26<00:00, 16.40it/s]
    Epoch 14/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.16it/s]
    

    ✅ Saved best model!
    Epoch 14/20 | Train 0.0391 | Val 0.0708 | Dice 0.9588 | IoU 0.9546 | LR 1.00e-04
    

    Epoch 15/20 [Train]: 100%|██████████| 2395/2395 [02:26<00:00, 16.39it/s]
    Epoch 15/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.19it/s]
    

    Epoch 15/20 | Train 0.0353 | Val 0.0842 | Dice 0.9559 | IoU 0.9528 | LR 1.00e-04
    

    Epoch 16/20 [Train]: 100%|██████████| 2395/2395 [02:26<00:00, 16.33it/s]
    Epoch 16/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.16it/s]
    

    ✅ Saved best model!
    Epoch 16/20 | Train 0.0344 | Val 0.0706 | Dice 0.9606 | IoU 0.9552 | LR 1.00e-04
    

    Epoch 17/20 [Train]: 100%|██████████| 2395/2395 [02:26<00:00, 16.36it/s]
    Epoch 17/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.19it/s]
    

    ✅ Saved best model!
    Epoch 17/20 | Train 0.0337 | Val 0.0756 | Dice 0.9608 | IoU 0.9563 | LR 1.00e-04
    

    Epoch 18/20 [Train]: 100%|██████████| 2395/2395 [02:26<00:00, 16.31it/s]
    Epoch 18/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.20it/s]
    

    Epoch 18/20 | Train 0.0303 | Val 0.0768 | Dice 0.9606 | IoU 0.9564 | LR 1.00e-04
    

    Epoch 19/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.44it/s]
    Epoch 19/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.20it/s]
    

    ✅ Saved best model!
    Epoch 19/20 | Train 0.0306 | Val 0.0649 | Dice 0.9625 | IoU 0.9573 | LR 1.00e-04
    

    Epoch 20/20 [Train]: 100%|██████████| 2395/2395 [02:25<00:00, 16.45it/s]
    Epoch 20/20 [Val]: 100%|██████████| 599/599 [00:53<00:00, 11.17it/s]
    

    Epoch 20/20 | Train 0.0285 | Val 0.0833 | Dice 0.9592 | IoU 0.9554 | LR 1.00e-04
    


```python
plot_res(train_losses,val_losses,val_dice,val_iou)
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_50_0.png)
    



```python
def show_samples(model, dataloader, device, num_samples=4, thr=0.5):
    model.eval()
    images, masks = next(iter(dataloader))  
    images = images.to(device)
    masks  = (masks > 0).float().to(device)  

    with torch.no_grad():
        outputs = model(images)              

        if outputs.shape[1] == 1: 
            probs = torch.sigmoid(outputs)
            preds = (probs > thr).float()
        else:                     
            probs = torch.softmax(outputs, dim=1)[:,1:2]
            preds = (probs > thr).float()

    images = images.cpu().permute(0,2,3,1).numpy()
    masks  = masks.cpu().squeeze().numpy()
    preds  = preds.cpu().squeeze().numpy()

    for i in range(num_samples):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow((images[i]*0.229+0.485).clip(0,1)) 
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(masks[i], cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(preds[i], cmap="gray")
        plt.title("Prediction")
        plt.axis("off")
        plt.show()



show_samples(model, val_loader, device, num_samples=5, thr=0.5)
```


    
![png](Kvasir-SEG_files/Kvasir-SEG_51_0.png)
    



    
![png](Kvasir-SEG_files/Kvasir-SEG_51_1.png)
    



    
![png](Kvasir-SEG_files/Kvasir-SEG_51_2.png)
    



    
![png](Kvasir-SEG_files/Kvasir-SEG_51_3.png)
    



    
![png](Kvasir-SEG_files/Kvasir-SEG_51_4.png)
    



```python

```
