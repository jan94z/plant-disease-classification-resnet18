if __name__ == '__main__':
    # imports
    import utils
    import os
    import shutil
    import argparse
    import torch
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    from torchmetrics import PrecisionRecallCurve
    from tqdm import tqdm
    from torch.nn.functional import softmax
    from skimage.io import imread
    import matplotlib.pyplot as plt

    # parsing
    _parser = argparse.ArgumentParser(description='Extracting command line arguments', add_help=True)
    _parser.add_argument('--t', '--training', action='store_const', const=True, help='No argument required')
    _parser.add_argument('--e', '--evaluation', action='store', help='Argument: Name of the model to evaluate')
    _parser.add_argument('--i', '--image', action='store', nargs='+', type=str, help='First argument: Name of the saved model. Second argument: Path to the image to be classified')
    parser = _parser.parse_args()

    classes = {0: 'healthy', 1: 'infected'}
    
    ######## TRAINING AND VALIDATION ########
    def training():
        
        # load specs
        config = utils.load_config('./config.yaml')
        training_path, validation_path = config['training_directory'], config['validation_directory']
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        epochs = config['epochs']
        model_name = config['model_name']
        pre_trained = config['pre_trained'] # flag to use pre-trained model or not
        if pre_trained: # use imagenet vaues
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else: # use calculated values of training data
            mean, std = config['mean'], config['std']
        model_path = f'./models/{model_name}'
        os.mkdir(model_path) # make directory to store data
        shutil.copyfile('./config.yaml', f'{model_path}/{model_name}_config.yaml') # make copy of the config file for reproduction

        # model
        net = resnet18(pre_trained)
        if pre_trained:
            for param in net.parameters():
                param.requires_grad = False
        fc_input = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2))

        transform = transforms.Compose( # data is already augmented, no need for further augmentation here
        [transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=mean, std=std)])
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        
        trainset = utils.imagedataset('./binary.csv', 'training', training_path, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        validset = utils.imagedataset('./binary.csv', 'validation', validation_path, transform=transform)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
        
        first_epoch = True # flag to create label and score tensors
        for epoch in range(epochs):
            os.mkdir(f'{model_path}/epoch{epoch+1}') # folder to store model of each epoch
            print(f'Epoch: {epoch+1}/{epochs}')
            # training
            print(f'Training...')
            net.train()
            first_batch = True
            running_train_loss = 0.0
            for inputs, labels in tqdm(trainloader):
                optimizer.zero_grad()
                outputs = net(inputs)
                scores = softmax(outputs, 1).data[:,1]
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)

                if first_batch:
                    truth = labels
                    prediction = scores
                    first_batch = False
                else:
                    truth = torch.hstack((truth, labels))
                    prediction = torch.hstack((prediction, scores))

            # validation
            print('Validation...')
            net.eval()
            first_batch = True
            running_valid_loss = 0.0
            with torch.no_grad():
                for vinputs, vlabels in tqdm(validloader):
                    voutputs = net(vinputs)
                    vloss = loss_function(voutputs, vlabels)
                    vscores = softmax(voutputs, 1).data[:,1]

                    running_valid_loss += vloss.item() * vinputs.size(0)

                    if first_batch:
                        vtruth = vlabels
                        vprediction = vscores
                        first_batch = False
                    else:
                        vtruth = torch.hstack((vtruth, vlabels))
                        vprediction = torch.hstack((vprediction, vscores))

            torch.save(net.state_dict(), f'{model_path}/epoch{epoch+1}/{model_name}{epoch+1}.pth')

            # calculate precision, recall, f1
            pr_curve = PrecisionRecallCurve(task='binary')
            (precision, recall, treshold), (vprecision, vrecall, vtreshold) = pr_curve(prediction, truth), pr_curve(vprediction, vtruth)
            treshold, vtreshold = torch.hstack((treshold, torch.tensor([torch.nan]))), torch.hstack((vtreshold, torch.tensor([torch.nan])))
            f1, vf1 = utils.f1(precision, recall), utils.f1(vprecision, vrecall)
                        
            train_df = pd.DataFrame({
                'train_precision': precision.numpy(),
                'train_recall': recall.numpy(),
                'train_f1': f1.numpy(),
                'train_treshold': treshold.numpy()
            })

            valid_df = pd.DataFrame({
                'valid_precision': vprecision.numpy(),
                'valid_recall': vrecall.numpy(),
                'valid_f1': vf1.numpy(),
                'valid_treshold': vtreshold.numpy()
            })
            train_df.to_csv(f'{model_path}/epoch{epoch+1}/{model_name}{epoch+1}_train.csv')
            valid_df.to_csv(f'{model_path}/epoch{epoch+1}/{model_name}{epoch+1}_valid.csv')
            utils.plot_prc(precision, recall, f'{model_name}{epoch+1} Training', f'{model_path}/epoch{epoch+1}/{model_name}{epoch+1}_train.jpeg')
            utils.plot_prc(vprecision, vrecall, f'{model_name}{epoch+1} Validation', f'{model_path}/epoch{epoch+1}/{model_name}{epoch+1}_valid.jpeg')
            
            loss_df = pd.DataFrame({
                'train_loss': [running_train_loss/len(trainset)],
                'valid_loss': [running_valid_loss/len(validset)]
            }, index=[epoch])
            if first_epoch:
                loss_df.to_csv(f'{model_path}/{model_name}_loss.csv', header=True)
                first_epoch = False
            else:
                loss_df.to_csv(f'{model_path}/{model_name}_loss.csv', mode='a', header=False)

            idx, vidx = torch.argmax(f1), torch.argmax(vf1)
            f1max, p, r, vf1max, vp, vr = f1[idx], precision[idx], recall[idx], vf1[vidx], vprecision[vidx], vrecall[vidx]

            print(f"Epoch finished.\nPerformance on training data: Best F1-score: {f1max} Precision at best F1-Score: {p} Recall at best F1-Score: {r}\nPerformance on validation data: Best F1-score: {vf1max} Precision at best F1-Score: {vp} Recall at best F1-Score: {vr}")
        
        loss_data = pd.read_csv(f'{model_path}/{model_name}_loss.csv')
        utils.plot_loss(loss_data['train_loss'], loss_data['valid_loss'], config['epochs'], f'{model_path}/{model_name}_loss.jpeg')
        
    ######## EVALUATION ########
    def evaluation():
        # load specs
        epoch = parser.e[-1]
        model = parser.e.rstrip(epoch)
        path = f'./models/{model}'
        
        config = utils.load_config(f'{path}/{model}_config.yaml')
        evaluation_path = config['evaluation_directory']
        batch_size = config['batch_size']
        pre_trained = config['pre_trained']
        if pre_trained:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std = config['mean'], config['std']
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((256, 256)), 
        transforms.Normalize(mean=mean, std=std)])
        
        evalset = utils.imagedataset('./binary.csv', 'evaluation', evaluation_path, transform=transform)
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=False)
        net = resnet18()
        if pre_trained:
            for param in net.parameters():
                param.requires_grad = False
        fc_input = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2))
        net.load_state_dict(torch.load(f'{path}/epoch{epoch}/{model}{epoch}.pth'))
        net.eval()

        first_batch = True
        with torch.no_grad():
            for einputs, elabels in tqdm(evalloader):
                eoutputs = net(einputs)
                escores = softmax(eoutputs, 1).data[:,1]

                if first_batch:
                    etruth = elabels
                    eprediction = escores
                    first_batch = False
                else:
                    etruth = torch.hstack((etruth, elabels))
                    eprediction = torch.hstack((eprediction, escores))
        
        pr_curve = PrecisionRecallCurve(task='binary')
        eprecision, erecall, etreshold = pr_curve(eprediction, etruth)
        etreshold = torch.hstack((etreshold, torch.tensor([torch.nan])))
        ef1 = utils.f1(eprecision, erecall)

        save_df = pd.DataFrame({
        'eval_precision': eprecision.numpy(),
        'eval_recall': erecall.numpy(),
        'eval_f1': ef1.numpy(),
        'eval_treshold': etreshold.numpy(),
         })
        save_df.to_csv(f"{path}/epoch{epoch}/{model}{epoch}_eval.csv")
        utils.plot_prc(eprecision, erecall, f"{model}{epoch} Evaluation", f"{path}/epoch{epoch}/{model}{epoch}_eval.jpeg")

        eidx = torch.argmax(ef1)
        ef1max, ep, er = ef1[eidx], eprecision[eidx], erecall[eidx]
        print(f"Evaluation finished.\nPerformance on evaluation data: Best F1-score: {ef1max} Precision at best F1-Score: {ep} Recall at best F1-Score: {er}")

    ######## CLASSIFICATION ########
    def classification():
        epoch = parser.i[0][-1]
        model = parser.i[0].rstrip(epoch)
        path = f'./models/{model}'
        image_path = parser.i[1]
        path = os.path.join('./models', model)
        config = utils.load_config(f'{path}/{model}_config.yaml')
        pre_trained = config['pre_trained']

        if pre_trained:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std = config['mean'], config['std']
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((256, 256)), 
        transforms.Normalize(mean=mean, std=std)])

        net = resnet18()
        if pre_trained:
            for param in net.parameters():
                param.requires_grad = False
        fc_input = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2))

        net.load_state_dict(torch.load(f'{path}/epoch{epoch}/{model}{epoch}.pth'))
        net.eval()

        image = imread(image_path)
        image_transformed = transform(image).unsqueeze(0)
        output = net(image_transformed)
        scores = softmax(output, 1).data.squeeze(0)
        plt.figure()
        plt.imshow(image)
        np.set_printoptions(suppress=True)
        plt.title(f'{classes[0]}: {scores.numpy()[0]:.10f} {classes[1]}: {scores.numpy()[1]:.10f}')
        plt.axis('off')
        plt.show()

    if parser.t:
        training()

    if parser.e:
        evaluation()

    if parser.i:
        classification()

