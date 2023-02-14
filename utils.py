import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread

def load_config(path):
    with open(path) as fp:
        file = yaml.safe_load(fp)
    return file

class imagedataset():
    def __init__(self, annotation, set, dir, transform=None):
        df = pd.read_csv(annotation)
        self.annotation = df[df.iloc[:, 2] == set].reset_index(drop=True) # filter annotation file for train, valid or eval labels
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        img_path = f'{self.dir}/{self.annotation.iloc[idx, 0]}'
        image = imread(img_path)
        label = self.annotation.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def f1(precision, recall, eps=0.00000000001):
    return (2*precision*recall)/(precision+recall+eps)

def plot_prc(precision, recall, name, path, title='Precision-Recall Curve'):
    idx = torch.argmax(f1(precision, recall))
    plt.figure()
    plt.plot(recall, precision, color='navy', linewidth='3')
    plt.plot(recall[idx], precision[idx], 'r*', label='Highest F1 Score')
    plt.title(f'{title} {name}')
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel('Recall')
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(path)

def plot_loss(train_loss, valid_loss, epochs, path, title='Validation and Training Loss per Epoch'):
    plt.figure()
    plt.plot(valid_loss, label='Validation')
    plt.plot(train_loss, label='Training')
    plt.xticks([x for x in range(epochs)], [x for x in range(1, epochs+1)])
    plt.title(title)
    plt.legend()
    plt.savefig(path)