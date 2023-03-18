# plant-disease-classification
## Objective
In this project I used PyTorch, Imagenet's ResNet18 and the PlantVillage dataset to obtain a network that is able to classify healthy and infected plant leaves. The dataset usually is used to built a multiclass model that treats each plant-disease pair as a single class. However, this time I wanted to build a binary classifier that just distinguishes between healthy and infected leaves.
## Usage
### Data
[Download the data (with augmentation)](https://data.mendeley.com/datasets/tywbtsjrjv/1) and unzip it. <br>
The folder "Plant_leave_diseases_dataset_with_augmentation" must be put into the folder [data](data) as it is.

[Article of the original dataset](https://arxiv.org/pdf/1511.08060v2.pdf) <br>
[Article of the augmented version of the dataset which I used](https://www.sciencedirect.com/science/article/abs/pii/S0045790619300023?via%3Dihub) <br>


### Initialize virtual environment and install packages
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
``` 
### [config.yaml](config.yaml)
```yaml
training_directory: ./data/training
validation_directory: ./data/validation
evaluation_directory: ./data/evaluation
seed: 1
mean: [0.4635, 0.4889, 0.4080]
std: [0.1755, 0.1484, 0.1919]

model_name: example
learning_rate: 0.001
epochs: 5
batch_size: 32
pre_trained: True
``` 
### Prepare the data
```bash
python3 prepare_data.py
```
The executed script does the following:
- Splits the data into training, validation and evaluation sets (60/20/20)
- Creates the annotation file 
- Calculates the mean and std of the training data

Mean and std are then printed and need to be copied to the config file if you decide to change the seed.

### Train the model
```bash
python3 binary.py --t
```
The executed part of the script does the following:
- trains the network
- validates the network after each epoch
- saves the state of the network as well as its performance on the training and validation data after each epoch
- keeps record of the loss per epoch and saves it

The files will be saved to [models](models), e.g. [models/example](models/example)

The name of the model and some hyperparameters can be changed in the config file.
### Evaluate the model
```bash
python3 binary.py --e example5
```
(number behind model name represents the epoch) <br>

The executed part of the script does the following:
- evaluates the chosen version of the model
- saves its performance on the evaluation data 

[models/example/epoch5](models/example/epoch5)
### Classify an image
```bash
python3 binary.py --i example5 ./data/evaluation/Blueberry___healthy/image14.JPG
```
![img1](https://user-images.githubusercontent.com/110723912/218749168-f6e21eca-b26c-479e-aa52-1d1f711e0c24.png)

```bash
python3 binary.py --i example5 ./data/evaluation/Potato___Late_blight/image150.JPG
```
![img2](https://user-images.githubusercontent.com/110723912/218749183-c93ff4a4-6325-4d3f-9ab0-5aee1ec54adf.png)

```bash
python3 binary.py --i example5 ./data/evaluation/Tomato___Spider_mitesTwo-spotted_spider_mite/image17.JPG
```
![img3](https://user-images.githubusercontent.com/110723912/218749186-b8edaa0c-8ab7-409c-afe5-5a0d7ca998a0.png)

