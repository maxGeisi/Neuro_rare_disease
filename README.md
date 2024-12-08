## Neuro Rare Disease Prediction 

This project demonstrates a simplified machine learning pipeline for predicting rare neurological diseases from patient data. It includes:

- Data Preprocessing: From raw patient records to training/validation/test splits.
- Model Training: A small neural network (MLP) trained to classify patients as diseased or not, based on phenotypes, genes, and a set of clinical features.
- Evaluation & Results: Confusion matrices, classification reports, and example predictions are provided in Jupyter notebooks for qualitative interpretation.


### Note 

Simplified Data & Pipeline:
This project uses a simplified dataset and approach. In a real scenario, one would probably:

- Fit transformations (e.g., scaling, encoding) on the training data only

- The Biological Data is probably more complex and woould require additional quality checks, and a changed model layout.


### Project Structure:

# Project Name

## Project Structure

```text
data/
├── raw/               # Contains the original input data (e.g., patients.csv).
├── processed/         # Will be populated with train.jsonl, val.jsonl, test.jsonl after preprocessing,
                       # and neuro_model.pt after training.
model/                 # Where trained model weights are saved.

scripts/
├── data_preprocessing.py  # Processes raw data into train, val, and test splits.
├── train_model.py         # Trains the neural network on the processed data and saves the model weights.
├── fetch_data.py          # From previous DB Integration


notebooks/
├── EDA.ipynb              # Exploratory Data Analysis of the processed dataset.
├── training_and_eval.ipynb # Initial evaluation of the trained model, including feature importances and validation metrics.
├── results.ipynb          # Qualitative results and predictions on the test set, confusion matrices, and
                           # example case studies for individual patients.

utils/                     # Utility scripts and configuration files (if any). Config from Previous DB Integration

requirements.txt           # Lists the Python dependencies to recreate the environment.

README.md                  # Project overview, setup instructions, and explanations (this file).
```

## Setup Instructions


```bash
git clone https://github.com/maxGeisi/Neuro_rare_disease.git
cd Neuro_rare_disease


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
 ```
### Process Data and train model
```bash
python scripts/data_preprocessing.py

python scripts/train_model.py

```
After training completes, a `neuro_model.pt` file will be saved in `data/model/.

## Viewing the Outputs

### 1. Exploratory Data Analysis (EDA)
Open `notebooks/EDA.ipynb` in Jupyter:

```bash
jupyter notebook notebooks/EDA.ipynb
```
### 2. Training and Evaluation
```bash
jupyter notebook notebooks/training_and_eval.ipynb

```
### 3. Qualitative Results
Open `notebooks/results.ipynb to inspect:


- The classification report and confusion matrix on the test set. 

* Predictions for individual patients, alongside their phenotypes and genes.


```bash
jupyter notebook notebooks/results.ipynb

```


