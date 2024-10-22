import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_openml

DATA_PATH = '../data/'
class MyDataSet(Dataset):
    def __init__(self, *args):
        self.args_list = args

    def __getitem__(self, index):
        return tuple(item[index] for item in self.args_list)

    def __len__(self):
        return self.args_list[0].shape[0]

if __name__ == '__main__':
    x, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False, data_home=DATA_PATH)

    minst_dataset = MyDataSet(x, y)
    batch = DataLoader(minst_dataset, shuffle=True, batch_size=32)

    for i, (mini_x, mini_y) in enumerate(batch):
        print(mini_x.shape)
        print(mini_y)
        print(i)