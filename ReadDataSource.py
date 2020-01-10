'''
Load dataset

author Gu Jiapan
'''

from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


class ReadDataSource(Dataset):
    def __init__(self, x_dir, y_file):
        # self.x_list = glob.glob(x_dir + '/*.png')
        self.x_dir = x_dir
        self.y_df = pd.read_csv(y_file)
        self.x_list = self.y_df['ID'].tolist()
        self.y_list = self.y_df['months'].tolist()
        # self.y_idx = sorted(list(set(self.y_list)))

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        x_ID = self.x_list[item]
        x_path_list = glob.glob(self.x_dir + '/' + str(x_ID) + '_*.png')
        x_path_list = sorted(list(set(x_path_list)))
        x = torch.zeros((4,48,48))
        for idx, x_path in enumerate(x_path_list):
            x_loc = os.path.split(x_path)[1].rsplit('_')[1].rsplit('.')[0]
            # print(x_loc)
            img = Image.open(x_path)
            img = transforms.Compose([
                transforms.Resize((48,48)),
                transforms.ToTensor()
                ])(img)
            x[int(x_loc),:,:] = img
        y = self.y_list[item]
        return(x, y)

if __name__ == "__main__":
    x_dir = '/data/image'
    y_file = '/data1/label.csv'
    tr = ReadDataSource(x_dir, y_file)
    trld = DataLoader(dataset=tr, batch_size=4, shuffle=False)
    for idx, (x, y) in enumerate(trld):
        print('########',idx)
        import pdb; pdb.set_trace()
        print('#######',idx)
