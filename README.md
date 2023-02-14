# plant-disease-classification
## Objective
...
## Usage
### Data
[Article of the original dataset](https://arxiv.org/pdf/1511.08060v2.pdf) <br>
[Article of the augmented version of the dataset which I used](https://www.sciencedirect.com/science/article/abs/pii/S0045790619300023?via%3Dihub) <br>
[Download the data (with augmentation)](https://data.mendeley.com/datasets/tywbtsjrjv/1) and unzip it <br>
The folder "Plant_leave_diseases_dataset_with_augmentation" must be put into the folder [data](data) as it is 
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
The executed part of script does the following:
- trains the network
- validates the network after each epoch
- saves the state of the network as well as its performance on the training and validation data after each epoch

The name of the model and some hyperparameters can be changed in the config file.
### Evaluate the model
```bash
python3 binary.py --e example5
```
(number behind model name represents the epoch) <br>

The executed part of script does the following:
- evaluates the chosen version of the model
- saves its performance on the evaluation data 
### Classify an image
```bash
python3 binary.py --i example5 .
```
