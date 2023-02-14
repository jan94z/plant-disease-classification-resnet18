# plant-disease-classification
## Objective
...
## Usage
...
### Data
Download the data [here](https://data.mendeley.com/datasets/tywbtsjrjv/1)
### Initialize virtual environment and install packages
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
``` 
### Prepare the data
[prepare_data.py]()
```bash
python3 prepare_data.py
```
### Train the model
[binary.py]()
```bash
python3 main.py --t
```
### Classify an image
[model] = the name of the saved model <br>
[img] = the path of the image to be classified
```bash
python3 binary.py --i [model] [img]
```
The repository comes with one trained model named 'example'.

[^reference]: 