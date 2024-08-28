
import os

import argparse
import json

import matplotlib as mpl
import matplotlib.pyplot as plt

FULL_PATH = '/mlx/users/zafar.takhirov/repo/ARC-AGI'  # Need this in case of a cluster submission which loses relative path information
DATA_PATH = os.path.join(FULL_PATH, 'third-party', 'ARC-AGI', 'data')
TRAIN_PATH = os.path.join(DATA_PATH, 'training')




# def parse_args():
#     parser = argparse.ArgumentParser(description='ARC-AGI Sandbox')
#     parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the data')
#     vis_parser = parser.add_argument_group('Visualization')
#     vis_parser.add_argument('--show', action='store_true', help='Display the images')


COLORS = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd']
CMAP = mpl.colors.ListedColormap(COLORS)

def visualize(data):
    '''Visualizes the ARC task

    Data should have the following structure:
        {
            'train': [  # Pattern to understand the task
                {'input': [...], 'output': [...]},
                {'input': [...], 'output': [...]},
                ...
            ],
            'test': [   # Pattern to predict
                {'input': [...], 'output': [...]},
                {'input': [...], 'output': [...]},
                ...
            ]
        }
    '''
    num_train = len(data['train'])
    num_test = len(data['test'])

    fig, ax = plt.subplots(num_train + num_test, 2, figsize=(2, num_train + num_test))

    for idx in range(num_train):
        # bounds = np.linspace(0, )
        # norm = mpl.colors.BoundaryNorm(bounds, CMAP.N)
        ax[idx, 0].imshow(data['train'][idx]['input'], cmap=CMAP)

    return fig, ax


all_files = os.listdir(TRAIN_PATH)
file_num = 0

with open(os.path.join(TRAIN_PATH, all_files[file_num])) as f:
    data = json.load(f)

fig, ax = visualize(data)
fig.show()
