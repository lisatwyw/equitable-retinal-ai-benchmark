import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import torch
import torch.nn.functional as F

class ResizePadv0:
    def __init__(self, size=448):
        self.size = size

    def __call__(self, x):
        h, w = x.shape[-2:]
        scale = self.size / max(h, w)
        h, w = round(h * scale), round(w * scale)

        x = F.interpolate(x.unsqueeze(0).float(), size=(h, w),
                          mode="bilinear", align_corners=False).squeeze(0)

        pad_h, pad_w = self.size - h, self.size - w
        x = F.pad(x, (pad_w//2, pad_w-pad_w//2,
                      pad_h//2, pad_h-pad_h//2))
        return x


class ResizePad:
    def __init__(self, size=448): self.size = size

    def __call__(self, x):
        h, w = x.shape[-2:]
        s = self.size / max(h, w)
        x = F.interpolate(x[None].float(), (round(h*s), round(w*s)),
                          mode="bilinear", align_corners=False)[0] / 255.
        x = F.pad(x, ((self.size-x.shape[-1])//2, (self.size-x.shape[-1]+1)//2,
                      (self.size-x.shape[-2])//2, (self.size-x.shape[-2]+1)//2))
        return (x - .5) / .5

'''
tf = T.Compose([
    T.Resize((r, r), antialias=True),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize((.5,)*3, (.5,)*3)
])
'''

class BRSETDataset(Dataset):
    def __init__(self, df, data_root, file_format, transform=None, task=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.task = task
        self.file_format = file_format

    '''
    self.image_paths = {
    row.image_id: next(
        os.path.join(data_root, f"{row.image_id}.{ext}")
        for ext in ("jpg", "jpeg", "png")
        if os.path.exists(os.path.join(data_root, f"{row.image_id}.{ext}"))
    )
    for _, row in self.df.iterrows()
    }    
    '''
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, f"{row.image_id}.{self.file_format}")
        x = read_image(img_path)
        
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] > 3:
            x = x[:3]

        if self.transform:
            x = self.transform(x)

        if self.task is None:
            return x, idx

        y = torch.tensor(float(row[self.task]), dtype=torch.float32)
        return x, y, idx


        