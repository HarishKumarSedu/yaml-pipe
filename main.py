from logger import log 
from common import read_yaml
from pathlib import Path 
import zipfile
import os 
import sys 
import json
import joblib 
import urllib.request as requests
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

config = read_yaml(Path('config.yaml'))
log.info(f'YAML Pipeline started: >>>>>>>>>>>>>>>>>>>>>')
# 1. create the artifacts directory 
os.makedirs(config.artifacts_root, exist_ok=True)
# I. Data ingestion 
data_ingestion = config.data_ingestion 
    # 1. Create Data ingestion root repository 
os.makedirs(data_ingestion.root_dir, exist_ok=True)
    # 2. Download Data 
filename , header = requests.urlretrieve(
    url=data_ingestion.data_url,
    filename=data_ingestion.data_local_file,
)
log.info(f'I. Data Ingestion : Data Downloaded with filename {filename} ')
    # 3. unzipping the data 
with zipfile.ZipFile(data_ingestion.data_local_file, 'r') as zip_ref :
    zip_ref.extractall(data_ingestion.data_unzip_dir)
    log.info('I. Data Ingestion : data unziped ')

# II. Data Validation 
data_validation = config.data_validation
    # 1. Create Data validation root repository 
os.makedirs(data_validation.root_dir, exist_ok=True)
# check the data file exists 
schema = read_yaml(Path('schema.yaml'))
log.info('II. Data validation : >>>>>>>>>>>>>>>>>>>>>')
log.info('data schema loaded ')
if os.path.exists(data_validation.data_local_file)  :
    data = pd.read_csv(data_validation.data_local_file)
    # check all the columns except the target 
    data_columns = data.columns[:-1]
    for column_name in data_columns :
        if column_name not in schema.COLUMNS.keys():
            log.info(f'column dosenot exists in dataset {column_name}')
            with open(data_validation.data_validation_status_file, 'w') as file :
                file.write('Valid : False')
        else:
            with open(data_validation.data_validation_status_file, 'w') as file :
                file.write('Valid : True')
    log.info('II. Data validation : data validated')


data_transformation = config.data_transformation
log.info('III. Data Transformation : >>>>>>>>>>>>>>>>>>>>>')
# III. Data Transformation 
os.makedirs(data_transformation.root_dir, exist_ok=True)
    # 1. create the data transformatin directory 
if os.path.exists(data_validation.data_local_file)  :
    data = pd.read_csv(data_validation.data_local_file)

    #2. check the the data is valid or not 
    with open( data_validation.data_validation_status_file,'r' ) as file :
        # read the data 
        data_validation_status = file.read()
        if  'True' in data_validation_status :
            train, test = train_test_split(data, train_size=0.7, random_state=2)
            train.to_csv(os.path.join(data_transformation.root_dir, "train.csv"),index = False)
            test.to_csv(os.path.join(data_transformation.root_dir, "test.csv"),index = False)
            log.info(f'Data splitted into the train and test \n test size {test.shape} train size {train.shape}')
        elif 'False':
            log.info('III. Data Transformation : Data is not validated')
        else:
            log.error('III. Data Transformation : Something wrong with the data')

        log.info('III. Data Transformation : data transformation completed')
# IV. Model Trainer Started 
model_trainer = config.model_trainer
log.info('IV. Model Trainer : >>>>>>>>>>>>>>>>>>>>>')
    # 1. create the directory for model trainer 
os.makedirs(model_trainer.root_dir,exist_ok=True)
    # 2. Load Model Prarams 
params = read_yaml(Path('params.yaml'))
log.info('IV. Model Trainer : model Parameters Loaded')
train_data = pd.read_csv(model_trainer.train_data_path)
test_data = pd.read_csv(model_trainer.test_data_path)
log.info('IV. Model Trainer : Train and Test Data Loaded')
train_x = train_data.drop([schema.TARGET], axis=1)
test_x = test_data.drop([schema.TARGET], axis=1)
train_y = train_data[[schema.TARGET]]
test_y = test_data[[schema.TARGET]]
log.info('IV. Model Trainer : Target and Raw Data seperated')
lr = ElasticNet(alpha=params.ElasticNet.alpha, l1_ratio=params.ElasticNet.l1_ratio, random_state=42)
lr.fit(train_x, train_y)
joblib.dump(lr, os.path.join(model_trainer.root_dir, model_trainer.model_name))
log.info('IV. Model Trainer : Model Training is completed')
# V. model Evaluation
log.info('V. Model Evaluation : >>>>>>>>>>>>>>>>>>>>>')
model_evaluation = config.model_evaluation
    # 1. create the data evaluvation directory 
os.makedirs(model_evaluation.root_dir, exist_ok=True)
test_data = pd.read_csv(model_evaluation.test_data_path)
test_x = test_data.drop([schema.TARGET], axis=1)
test_y = test_data[[schema.TARGET]]
log.info('V. Model Evaluation : Test Data Loaded')
model = joblib.load(model_evaluation.model_path)
log.info('V. Model Evaluation : Binary Model Loaded')
y_pred = model.predict(test_x)

r2  = r2_score(test_y, y_pred)
mse  = mean_squared_error(test_y, y_pred)
metrics = {
    "r2":r2,
    "mse":mse
}
log.info(f'V. Model Evaluation : Model Metrics >  {metrics}')

with open(model_evaluation.metric_file_name, 'w') as file :
    json.dump(metrics, file)
log.info(f'V. Model Evaluation : Model Evaluation Completed')
    