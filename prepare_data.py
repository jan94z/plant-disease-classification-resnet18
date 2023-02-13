if __name__ == '__main__':
    import utils
    import os
    import shutil
    import torch
    import numpy as np
    import pandas as pd
    import torchvision.transforms as transforms
    from tqdm import tqdm
    from skimage.io import imread

    config = utils.load_config('./config.yaml')
    training_path = config['training_directory']
    validation_path = config['validation_directory']
    evaluation_path = config['evaluation_directory']

    ######## split data -> 60/20/20 ########
    for folder in [training_path, validation_path, evaluation_path]:
        os.mkdir(folder)

    origin_dir = './data/Plant_leave_diseases_dataset_with_augmentation'

    # found a weird image with 4 channels displaying something different in the dataset -> leave such images behind
    imgs_to_delete = []
    for folder in os.listdir(origin_dir):
         for img in os.listdir(os.path.join(origin_dir, folder)):
             image = imread(os.path.join(origin_dir, folder, img))
             if image.shape[-1] == 4:
                 imgs_to_delete.append(os.path.join(origin_dir, folder, img))
         os.rename(os.path.join(origin_dir, folder), os.path.join(origin_dir, folder).replace(' ', '').replace('(', '').replace(')', ''))
    for img in imgs_to_delete:
        print('Image deleted: ', img)
        os.remove(img)

    for folder in os.listdir(origin_dir):
        if 'Background' not in folder:
            for f in [training_path, validation_path, evaluation_path]:
                os.mkdir(os.path.join(f, folder))
            images = os.listdir(os.path.join(origin_dir, folder))
            indeces = np.arange(0, len(images), step=1, dtype=int)
            np.random.seed(config['seed'])
            np.random.shuffle(indeces)
            # move validation images
            for idx in indeces[0:round(0.2*len(indeces))]:
                shutil.move(os.path.join(origin_dir, folder, images[idx]), os.path.join(validation_path, folder, images[idx]).replace(' ', '').replace('(', '').replace(')', ''))
            # move evaluation images
            for idx in indeces[round(0.2*len(indeces)):round(0.4*len(indeces))]:
                shutil.move(os.path.join(origin_dir, folder, images[idx]), os.path.join(evaluation_path, folder, images[idx]).replace(' ', '').replace('(', '').replace(')', ''))
            # move training images
            for idx in indeces[round(0.4*len(indeces)):]:
                shutil.move(os.path.join(origin_dir, folder, images[idx]), os.path.join(training_path, folder, images[idx]).replace(' ', '').replace('(', '').replace(')', ''))

    ######## create annotation file ########
    path, binary_classes, multi_classes, set = [], [], [], []
    class_dict = {}

    for dir in [training_path, validation_path, evaluation_path]:
        for idx, folder in enumerate(sorted(os.listdir(dir))):
            class_dict['folder'] = idx
            for file in sorted(os.listdir(os.path.join(dir, folder))):
                # path
                path.append(os.path.join(folder, file))
                # labels
                if 'healthy' in folder:
                    binary_classes.append(0) # healthy = label 0
                else:
                    binary_classes.append(1) # infected = label 1
                multi_classes.append(idx)
                # set
                if dir == training_path:
                    set.append('training')
                elif dir == validation_path:
                    set.append('validation')
                elif dir == evaluation_path:
                    set.append('evaluation')
    
    binary = pd.DataFrame({
        'path': path,
        'classes': binary_classes,
        'set': set
    })
    binary = binary.sort_values('path')
    binary.to_csv('binary.csv', index=False, header=False)

    # could be included if the task was not binary but multi-class
    """
    multi = pd.DataFrame({
    'path': path,
    'classes': multi_classes,
    'set': set
    })
    multi = multi.sort_values('path')
    multi.to_csv('multi.csv', index=False, header=False)
    """

    ######## calculate mean and std ########
    trainset = utils.imagedataset(annotation='./binary.csv', set= 'training', dir=training_path, 
    transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)

    # calc mean and std of all pixels per image channel-wise, sum up means and stds of all images, divide by amount of images
    first = True
    for batch, _ in tqdm(trainloader):
        batch = batch
        batch = batch.view(batch.size(0), batch.size(1), -1)
        if first:
            n_images = batch.size(0) 
            mean = batch.mean(2).sum(0)
            std = batch.std(2).sum(0)
            first = False
        else:
            n_images += batch.size(0)
            mean += batch.mean(2).sum(0)
            std += batch.std(2).sum(0)
    mean /= n_images
    std /= n_images
    print('mean: ', mean, 'std: ', std) # -> put to config file