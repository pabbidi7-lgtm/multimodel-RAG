<!-- image -->

### MODEL TRAINING .ipynb

Model Training

**1.1 Import Data and Required Packages**

**Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.**

In [6]:

*# Basic Import*

**import** numpy **as** np

**import** pandas **as** pd

**import** matplotlib.pyplot **as** plt

**import** seaborn **as** sns

*# Modelling*

**from** sklearn.metrics **import** mean\_squared\_error, r2\_score

**from** sklearn.neighbors **import** KNeighborsRegressor

**from** sklearn.tree **import** DecisionTreeRegressor

**from** sklearn.ensemble **import** RandomForestRegressor,AdaBoostRegressor

**from** sklearn.svm **import** SVR

**from** sklearn.linear\_model **import** LinearRegression, Ridge,Lasso

**from** sklearn.metrics **import** r2\_score, mean\_absolute\_error, mean\_squared\_error

**from** sklearn.model\_selection **import** RandomizedSearchCV

**from** catboost **import** CatBoostRegressor

**from** xgboost **import** XGBRegressor

**import** warnings

**Import the CSV Data as Pandas DataFrame**

In [8]:

df **=** pd **.** read\_csv('data/stud.csv')

**Show Top 5 Records**

In [9]:

df **.** head()

Out[9]:

|       | **gender**   | **race\_ethnicity**   | **parental\_level\_of\_education**   | **lunch**    | **test\_preparation\_course**   |   **math\_score** |   **reading\_score** |   **writing\_score** |
|-------|--------------|-----------------------|--------------------------------------|--------------|---------------------------------|-------------------|----------------------|----------------------|
| **0** | female       | group B               | bachelor's degree                    | standard     | none                            |                72 |                   72 |                   74 |
| **1** | female       | group C               | some college                         | standard     | completed                       |                69 |                   90 |                   88 |
| **2** | female       | group B               | master's degree                      | standard     | none                            |                90 |                   95 |                   93 |
| **3** | male         | group A               | associate's degree                   | free/reduced | none                            |                47 |                   57 |                   44 |
| **4** | male         | group C               | some college                         | standard     | none                            |                76 |                   78 |                   75 |

**Preparing X and Y variables**

In [10]:

X **=** df **.** drop(columns **=** ['math\_score'],axis **=** 1)

In [11]:

X **.** head()

Out[11]:

|       | **gender**   | **race\_ethnicity**   | **parental\_level\_of\_education**   | **lunch**    | **test\_preparation\_course**   |   **reading\_score** |   **writing\_score** |
|-------|--------------|-----------------------|--------------------------------------|--------------|---------------------------------|----------------------|----------------------|
| **0** | female       | group B               | bachelor's degree                    | standard     | none                            |                   72 |                   74 |
| **1** | female       | group C               | some college                         | standard     | completed                       |                   90 |                   88 |
| **2** | female       | group B               | master's degree                      | standard     | none                            |                   95 |                   93 |
| **3** | male         | group A               | associate's degree                   | free/reduced | none                            |                   57 |                   44 |
| **4** | male         | group C               | some college                         | standard     | none                            |                   78 |                   75 |

In [14]:

print("Categories in 'gender' variable:     ",end **=** " " )

print(df['gender'] **.** unique())

print("Categories in 'race\_ethnicity' variable:  ",end **=** " ")

print(df['race\_ethnicity'] **.** unique())

print("Categories in'parental level of education' variable:",end **=** " " )

print(df['parental\_level\_of\_education'] **.** unique())

print("Categories in 'lunch' variable:     ",end **=** " " )

print(df['lunch'] **.** unique())

print("Categories in 'test preparation course' variable:     ",end **=** " " )

print(df['test\_preparation\_course'] **.** unique())

Categories in 'gender' variable:      ['female' 'male']

Categories in 'race\_ethnicity' variable:   ['group B' 'group C' 'group A' 'group D' 'group E']

Categories in'parental level of education' variable: ["bachelor's degree" 'some college' "master's degree" "associate's degree"

'high school' 'some high school']

Categories in 'lunch' variable:      ['standard' 'free/reduced']

Categories in 'test preparation course' variable:      ['none' 'completed']

In [12]:

y **=** df['math\_score']

In [13]:

y

Out[13]:

0      72

1      69

2      90

3      47

4      76

..

995    88

996    62

997    59

998    68

999    77

Name: math\_score, Length: 1000, dtype: int64

In [15]:

*# Create Column Transformer with 3 types of transformers*

num\_features **=** X **.** select\_dtypes(exclude **=** "object") **.** columns

cat\_features **=** X **.** select\_dtypes(include **=** "object") **.** columns

**from** sklearn.preprocessing **import** OneHotEncoder, StandardScaler

**from** sklearn.compose **import** ColumnTransformer

numeric\_transformer **=** StandardScaler()

oh\_transformer **=** OneHotEncoder()

preprocessor **=** ColumnTransformer(

[

("OneHotEncoder", oh\_transformer, cat\_features),

("StandardScaler", numeric\_transformer, num\_features),

]

)

In [16]:

X **=** preprocessor **.** fit\_transform(X)

In [18]:

X **.** shape

Out[18]:

(1000, 19)

In [21]:

*# separate dataset into train and test*

**from** sklearn.model\_selection **import** train\_test\_split

X\_train, X\_test, y\_train, y\_test **=** train\_test\_split(X,y,test\_size **=** 0.2,random\_state **=** 42)

X\_train **.** shape, X\_test **.** shape

Out[21]:

((800, 19), (200, 19))

**Create an Evaluate Function to give all metrics after model Training**

In [22]:

**def** evaluate\_model(true, predicted):

mae **=** mean\_absolute\_error(true, predicted)

mse **=** mean\_squared\_error(true, predicted)

rmse **=** np **.** sqrt(mean\_squared\_error(true, predicted))

r2\_square **=** r2\_score(true, predicted)

**return** mae, rmse, r2\_square

In [23]:

models **=** {

"Linear Regression": LinearRegression(),

"Lasso": Lasso(),

"Ridge": Ridge(),

"K-Neighbors Regressor": KNeighborsRegressor(),

"Decision Tree": DecisionTreeRegressor(),

"Random Forest Regressor": RandomForestRegressor(),

"XGBRegressor": XGBRegressor(),

"CatBoosting Regressor": CatBoostRegressor(verbose **=False** ),

"AdaBoost Regressor": AdaBoostRegressor()

}

model\_list **=** []

r2\_list **=** []

**for** i **in** range(len(list(models))):

model **=** list(models **.** values())[i]

model **.** fit(X\_train, y\_train) *# Train model*

*# Make predictions*

y\_train\_pred **=** model **.** predict(X\_train)

y\_test\_pred **=** model **.** predict(X\_test)

*# Evaluate Train and Test dataset*

model\_train\_mae , model\_train\_rmse, model\_train\_r2 **=** evaluate\_model(y\_train, y\_train\_pred)

model\_test\_mae , model\_test\_rmse, model\_test\_r2 **=** evaluate\_model(y\_test, y\_test\_pred)

print(list(models **.** keys())[i])

model\_list **.** append(list(models **.** keys())[i])

print('Model performance for Training set')

print("- Root Mean Squared Error: {:.4f}" **.** format(model\_train\_rmse))

print("- Mean Absolute Error: {:.4f}" **.** format(model\_train\_mae))

print("- R2 Score: {:.4f}" **.** format(model\_train\_r2))

print('----------------------------------')

print('Model performance for Test set')

print("- Root Mean Squared Error: {:.4f}" **.** format(model\_test\_rmse))

print("- Mean Absolute Error: {:.4f}" **.** format(model\_test\_mae))

print("- R2 Score: {:.4f}" **.** format(model\_test\_r2))

r2\_list **.** append(model\_test\_r2)

print('=' ***** 35)

print('\n')

Linear Regression

Model performance for Training set

- Root Mean Squared Error: 5.3243

- Mean Absolute Error: 4.2671

- R2 Score: 0.8743

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 5.3960

- Mean Absolute Error: 4.2158

- R2 Score: 0.8803

===================================

Lasso

Model performance for Training set

- Root Mean Squared Error: 6.5938

- Mean Absolute Error: 5.2063

- R2 Score: 0.8071

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 6.5197

- Mean Absolute Error: 5.1579

- R2 Score: 0.8253

===================================

Ridge

Model performance for Training set

- Root Mean Squared Error: 5.3233

- Mean Absolute Error: 4.2650

- R2 Score: 0.8743

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 5.3904

- Mean Absolute Error: 4.2111

- R2 Score: 0.8806

===================================

K-Neighbors Regressor

Model performance for Training set

- Root Mean Squared Error: 5.7077

- Mean Absolute Error: 4.5167

- R2 Score: 0.8555

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 7.2530

- Mean Absolute Error: 5.6210

- R2 Score: 0.7838

===================================

Decision Tree

Model performance for Training set

- Root Mean Squared Error: 0.2795

- Mean Absolute Error: 0.0187

- R2 Score: 0.9997

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 7.6371

- Mean Absolute Error: 6.0250

- R2 Score: 0.7603

===================================

Random Forest Regressor

Model performance for Training set

- Root Mean Squared Error: 2.2851

- Mean Absolute Error: 1.8253

- R2 Score: 0.9768

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 6.0959

- Mean Absolute Error: 4.7194

- R2 Score: 0.8473

===================================

XGBRegressor

Model performance for Training set

- Root Mean Squared Error: 0.9087

- Mean Absolute Error: 0.6148

- R2 Score: 0.9963

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 6.5889

- Mean Absolute Error: 5.0844

- R2 Score: 0.8216

===================================

CatBoosting Regressor

Model performance for Training set

- Root Mean Squared Error: 3.0427

- Mean Absolute Error: 2.4054

- R2 Score: 0.9589

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 6.0086

- Mean Absolute Error: 4.6125

- R2 Score: 0.8516

===================================

AdaBoost Regressor

Model performance for Training set

- Root Mean Squared Error: 5.7843

- Mean Absolute Error: 4.7564

- R2 Score: 0.8516

----------------------------------

Model performance for Test set

- Root Mean Squared Error: 6.0447

- Mean Absolute Error: 4.6813

- R2 Score: 0.8498

===================================

**Results**

In [24]:

pd **.** DataFrame(list(zip(model\_list, r2\_list)), columns **=** ['Model Name', 'R2\_Score']) **.** sort\_values(by **=** ["R2\_Score"],ascending **=False** )

Out[24]:

|       | **Model Name**          |   **R2\_Score** |
|-------|-------------------------|-----------------|
| **2** | Ridge                   |        0.880593 |
| **0** | Linear Regression       |        0.880345 |
| **7** | CatBoosting Regressor   |        0.851632 |
| **8** | AdaBoost Regressor      |        0.849847 |
| **5** | Random Forest Regressor |        0.847291 |
| **1** | Lasso                   |        0.82532  |
| **6** | XGBRegressor            |        0.821589 |
| **3** | K-Neighbors Regressor   |        0.783813 |
| **4** | Decision Tree           |        0.760313 |

**Linear Regression**

In [25]:

lin\_model **=** LinearRegression(fit\_intercept **=True** )

lin\_model **=** lin\_model **.** fit(X\_train, y\_train)

y\_pred **=** lin\_model **.** predict(X\_test)

score **=** r2\_score(y\_test, y\_pred) ***** 100

print(" Accuracy of the model is %.2f" **%sc** ore **)**

Accuracy of the model is 88.03

**Plot y\_pred and y\_test**

In [26]:

plt **.** scatter(y\_test,y\_pred);

plt **.** xlabel('Actual');

plt **.** ylabel('Predicted');

<!-- image -->

In [27]:

sns **.** regplot(x **=** y\_test,y **=** y\_pred,ci **=None** ,color **=** 'red');

<!-- image -->

**Difference between Actual and Predicted Values**

In [28]:

pred\_df **=** pd **.** DataFrame({'Actual Value':y\_test,'Predicted Value':y\_pred,'Difference':y\_test **-** y\_pred})

pred\_df

Out[28]:

|         | **Actual Value**   | **Predicted Value**   | **Difference**   |
|---------|--------------------|-----------------------|------------------|
| **521** | 91                 | 76.507812             | 14.492188        |
| **737** | 53                 | 58.953125             | -5.953125        |
| **740** | 80                 | 76.960938             | 3.039062         |
| **660** | 74                 | 76.757812             | -2.757812        |
| **411** | 84                 | 87.539062             | -3.539062        |
| **...** | ...                | ...                   | ...              |
| **408** | 52                 | 43.546875             | 8.453125         |
| **332** | 62                 | 62.031250             | -0.031250        |
| **208** | 74                 | 67.976562             | 6.023438         |
| **613** | 65                 | 67.132812             | -2.132812        |
| **78**  | 61                 | 62.492188             | -1.492188        |

200 rows × 3 columns

In [ ]:

AS PER THE IMAGE OF WORKFLOW WE CREATED THE NOTEBOOK FOLDER AND AGAIN CREATED THE FOLDER CALLED CATBOOST\_INFO AGAIN WE CREATED THE DATA FOLDER SO THIS ABOVE CODE IS BELONG TO THE DATA FOLDER

AFTER THIS WE ARE CREATING THE SRC FOLDER AND IN THAT WE HAVE ADDED THE COMPONENTS FOLDER IN THAT WE CREATED THE FILES CALLED data\_ingestion.py, data\_transformation.py and model\_training.py **DATA INGESTION CODE:**

import os

import sys

from src.exception import CustomException

from src.logger import logging

import pandas as pd

from sklearn.model\_selection import train\_test\_split

from dataclasses import dataclass

from src.components.data\_transformation import DataTransformation

from src.components.data\_transformation import DataTransformationConfig

from src.components.model\_trainer import ModelTrainerConfig

from src.components.model\_trainer import ModelTrainer

@dataclass

class DataIngestionConfig:

train\_data\_path: str=os.path.join('artifacts',"train.csv")

test\_data\_path: str=os.path.join('artifacts',"test.csv")

raw\_data\_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:

def \_\_init\_\_(self):

self.ingestion\_config=DataIngestionConfig()

def initiate\_data\_ingestion(self):

logging.info("Entered the data ingestion method or component")

try:

df=pd.read\_csv('notebook\data\stud.csv')

logging.info('Read the dataset as dataframe')

os.makedirs(os.path.dirname(self.ingestion\_config.train\_data\_path),exist\_ok=True)

df.to\_csv(self.ingestion\_config.raw\_data\_path,index=False,header=True)

logging.info("Train test split initiated")

train\_set,test\_set=train\_test\_split(df,test\_size=0.2,random\_state=42)

train\_set.to\_csv(self.ingestion\_config.train\_data\_path,index=False,header=True)

test\_set.to\_csv(self.ingestion\_config.test\_data\_path,index=False,header=True)

logging.info("Inmgestion of the data iss completed")

return(

self.ingestion\_config.train\_data\_path,

self.ingestion\_config.test\_data\_path

)

except Exception as e:

raise CustomException(e,sys)

if \_\_name\_\_=="\_\_main\_\_":

obj=DataIngestion()

train\_data,test\_data=obj.initiate\_data\_ingestion()

data\_transformation=DataTransformation()

train\_arr,test\_arr,\_=data\_transformation.initiate\_data\_transformation(train\_data,test\_data)

modeltrainer=ModelTrainer()

print(modeltrainer.initiate\_model\_trainer(train\_arr,test\_arr))

**DATA TRANSFORMAATION CODE:**

import sys

from dataclasses import dataclass

import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException

from src.logger import logging

import os

from src.utils import save\_object

@dataclass

class DataTransformationConfig:

preprocessor\_obj\_file\_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:

def \_\_init\_\_(self):

self.data\_transformation\_config=DataTransformationConfig()

def get\_data\_transformer\_object(self):

'''

This function si responsible for data trnasformation

'''

try:

numerical\_columns = ["writing\_score", "reading\_score"]

categorical\_columns = [

"gender",

"race\_ethnicity",

"parental\_level\_of\_education",

"lunch",

"test\_preparation\_course",

]

num\_pipeline= Pipeline(

steps=[

("imputer",SimpleImputer(strategy="median")),

("scaler",StandardScaler())

]

)

cat\_pipeline=Pipeline(

steps=[

("imputer",SimpleImputer(strategy="most\_frequent")),

("one\_hot\_encoder",OneHotEncoder()),

("scaler",StandardScaler(with\_mean=False))

]

)

logging.info(f"Categorical columns: {categorical\_columns}")

logging.info(f"Numerical columns: {numerical\_columns}")

preprocessor=ColumnTransformer(

[

("num\_pipeline",num\_pipeline,numerical\_columns),

("cat\_pipelines",cat\_pipeline,categorical\_columns)

]

)

return preprocessor

except Exception as e:

raise CustomException(e,sys)

def initiate\_data\_transformation(self,train\_path,test\_path):

try:

train\_df=pd.read\_csv(train\_path)

test\_df=pd.read\_csv(test\_path)

logging.info("Read train and test data completed")

logging.info("Obtaining preprocessing object")

preprocessing\_obj=self.get\_data\_transformer\_object()

target\_column\_name="math\_score"

numerical\_columns = ["writing\_score", "reading\_score"]

input\_feature\_train\_df=train\_df.drop(columns=[target\_column\_name],axis=1)

target\_feature\_train\_df=train\_df[target\_column\_name]

input\_feature\_test\_df=test\_df.drop(columns=[target\_column\_name],axis=1)

target\_feature\_test\_df=test\_df[target\_column\_name]

logging.info(

f"Applying preprocessing object on training dataframe and testing dataframe."

)

input\_feature\_train\_arr=preprocessing\_obj.fit\_transform(input\_feature\_train\_df)

input\_feature\_test\_arr=preprocessing\_obj.transform(input\_feature\_test\_df)

train\_arr = np.c\_[

input\_feature\_train\_arr, np.array(target\_feature\_train\_df)

]

test\_arr = np.c\_[input\_feature\_test\_arr, np.array(target\_feature\_test\_df)]

logging.info(f"Saved preprocessing object.")

save\_object(

file\_path=self.data\_transformation\_config.preprocessor\_obj\_file\_path,

obj=preprocessing\_obj

)

return (

train\_arr,

test\_arr,

self.data\_transformation\_config.preprocessor\_obj\_file\_path,

)

except Exception as e:

raise CustomException(e,sys)

**MODEL TRAINING CODE:**

import os

import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (

AdaBoostRegressor,

GradientBoostingRegressor,

RandomForestRegressor,

)

from sklearn.linear\_model import LinearRegression

from sklearn.metrics import r2\_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.exception import CustomException

from src.logger import logging

from src.utils import save\_object,evaluate\_models

@dataclass

class ModelTrainerConfig:

trained\_model\_file\_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:

def \_\_init\_\_(self):

self.model\_trainer\_config=ModelTrainerConfig()

def initiate\_model\_trainer(self,train\_array,test\_array):

try:

logging.info("Split training and test input data")

X\_train,y\_train,X\_test,y\_test=(

train\_array[:,:-1],

train\_array[:,-1],

test\_array[:,:-1],

test\_array[:,-1]

)

models = {

"Random Forest": RandomForestRegressor(),

"Decision Tree": DecisionTreeRegressor(),

"Gradient Boosting": GradientBoostingRegressor(),

"Linear Regression": LinearRegression(),

"XGBRegressor": XGBRegressor(),

"CatBoosting Regressor": CatBoostRegressor(verbose=False),

"AdaBoost Regressor": AdaBoostRegressor(),

}

params={

"Decision Tree": {

'criterion':['squared\_error', 'friedman\_mse', 'absolute\_error', 'poisson'],

# 'splitter':['best','random'],

# 'max\_features':['sqrt','log2'],

},

"Random Forest":{

# 'criterion':['squared\_error', 'friedman\_mse', 'absolute\_error', 'poisson'],

# 'max\_features':['sqrt','log2',None],

'n\_estimators': [8,16,32,64,128,256]

},

"Gradient Boosting":{

# 'loss':['squared\_error', 'huber', 'absolute\_error', 'quantile'],

'learning\_rate':[.1,.01,.05,.001],

'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],

# 'criterion':['squared\_error', 'friedman\_mse'],

# 'max\_features':['auto','sqrt','log2'],

'n\_estimators': [8,16,32,64,128,256]

},

"Linear Regression":{},

"XGBRegressor":{

'learning\_rate':[.1,.01,.05,.001],

'n\_estimators': [8,16,32,64,128,256]

},

"CatBoosting Regressor":{

'depth': [6,8,10],

'learning\_rate': [0.01, 0.05, 0.1],

'iterations': [30, 50, 100]

},

"AdaBoost Regressor":{

'learning\_rate':[.1,.01,0.5,.001],

# 'loss':['linear','square','exponential'],

'n\_estimators': [8,16,32,64,128,256]

}

}

model\_report:dict=evaluate\_models(X\_train=X\_train,y\_train=y\_train,X\_test=X\_test,y\_test=y\_test,

models=models,param=params)

## To get best model score from dict

best\_model\_score = max(sorted(model\_report.values()))

## To get best model name from dict

best\_model\_name = list(model\_report.keys())[

list(model\_report.values()).index(best\_model\_score)

]

best\_model = models[best\_model\_name]

if best\_model\_score&lt;0.6:

raise CustomException("No best model found")

logging.info(f"Best found model on both training and testing dataset")

save\_object(

file\_path=self.model\_trainer\_config.trained\_model\_file\_path,

obj=best\_model

)

predicted=best\_model.predict(X\_test)

r2\_square = r2\_score(y\_test, predicted)

return r2\_square

except Exception as e:

raise CustomException(e,sys)

- AFTER THAT WE HAVE CREATED THE PIPELINE FOLDER AND CREATED AGAIN 3 FILES exception.py, logger.py, and utils.py **EXCEPTION CODE:**

import sys

from src.logger import logging

def error\_message\_detail(error,error\_detail:sys):

\_,\_,exc\_tb=error\_detail.exc\_info()

file\_name=exc\_tb.tb\_frame.f\_code.co\_filename

error\_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(

file\_name,exc\_tb.tb\_lineno,str(error))

return error\_message

class CustomException(Exception):

def \_\_init\_\_(self,error\_message,error\_detail:sys):

super().\_\_init\_\_(error\_message)

self.error\_message=error\_message\_detail(error\_message,error\_detail=error\_detail)

def \_\_str\_\_(self):

return self.error\_message

**LOGGER CODE:**

import logging

import os

from datetime import datetime

LOG\_FILE=f"{datetime.now().strftime('%m\_%d\_%Y\_%H\_%M\_%S')}.log"

logs\_path=os.path.join(os.getcwd(),"logs",LOG\_FILE)

os.makedirs(logs\_path,exist\_ok=True)

LOG\_FILE\_PATH=os.path.join(logs\_path,LOG\_FILE)

logging.basicConfig(

filename=LOG\_FILE\_PATH,

format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",

level=logging.INFO,

)

**UTILS CODE:**

import os

import sys

import numpy as np

import pandas as pd

import dill

from sklearn.metrics import r2\_score

from sklearn.model\_selection import GridSearchCV

from src.exception import CustomException

def save\_object(file\_path, obj):

try:

dir\_path = os.path.dirname(file\_path)

os.makedirs(dir\_path, exist\_ok=True)

with open(file\_path, "wb") as file\_obj:

dill.dump(obj, file\_obj)

except Exception as e:

raise CustomException(e, sys)

def evaluate\_models(X\_train, y\_train,X\_test,y\_test,models,param):

try:

report = {}

for i in range(len(list(models))):

model = list(models.values())[i]

para=param[list(models.keys())[i]]

gs = GridSearchCV(model,para,cv=3)

gs.fit(X\_train,y\_train)

model.set\_params(**gs.best\_params\_)

model.fit(X\_train,y\_train)

#model.fit(X\_train, y\_train)  # Train model

y\_train\_pred = model.predict(X\_train)

y\_test\_pred = model.predict(X\_test)

train\_model\_score = r2\_score(y\_train, y\_train\_pred)

test\_model\_score = r2\_score(y\_test, y\_test\_pred)

report[list(models.keys())[i]] = test\_model\_score

return report

except Exception as e:

raise CustomException(e, sys)

def load\_object(file\_path):

try:

with open(file\_path, "rb") as file\_obj:

return dill.load(file\_obj)

except Exception as e:

raise CustomException(e, sys)

SO AFTER THE MODEL TRAINING WAS DONE IN THE DATA FOLDER WE WILL PERFORM THE COMPONENTS FOLDER AND PERFORM THE ABOVE MENTIONED data\_ingestion, data\_transformation, model\_trainer.py SO WHILE PERFORMING THESE THE CODE PART IS GIVING THE data.csv, model.pkl, preprocessor.pkl, test.csv, train.csv SO THESE 5 FILES ARE BELONGS TO THE artifacts
BASICALLY THE ARTIFACTS FORLDER WILL GET THE INDIVIDUAL FOLDERS THAT WE ARE PERFORMING FROM THE DATA FOLDER AND COMPONENTS FOLDER AS PER THE FLOW WE ARE UPDATING IN THE PIPELINE FOLDER 

SIMULTANIOUSLY WE ARE KEEP ON UPDATING IN THE templates FOLDER WE HAVE THE Dockerfile, README.md,  app.py, requirements.txt, setup.py

**IN THE DOCKER FILE WE HAVE** code:

FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN apt update -y &amp;&amp; apt install awscli -y

RUN apt-get update &amp;&amp; apt-get install ffmpeg libsm6 libxext6 unzip -y &amp;&amp; pip install -r requirements.txt

CMD ["python3", "app.py"]

**IN THE README.MD:**

## End to End MAchine Learning Project

1. Docker Build checked

2. Github Workflow

3. Iam User In AWS

## Docker Setup In EC2 commands to be Executed

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

## Configure EC2 as self-hosted runner:

## Setup github secrets:

AWS\_ACCESS\_KEY\_ID=

AWS\_SECRET\_ACCESS\_KEY=

AWS\_REGION = us-east-1

AWS\_ECR\_LOGIN\_URI = demo&gt;&gt;  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR\_REPOSITORY\_NAME = simple-app

**IN THE app.py:**

from flask import Flask,request,render\_template

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict\_pipeline import CustomData,PredictPipeline

application=Flask(\_\_name\_\_)

app=application

## Route for a home page

@app.route('/')

def index():

return render\_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])

def predict\_datapoint():

if request.method=='GET':

return render\_template('home.html')

else:

data=CustomData(

gender=request.form.get('gender'),

race\_ethnicity=request.form.get('ethnicity'),

parental\_level\_of\_education=request.form.get('parental\_level\_of\_education'),

lunch=request.form.get('lunch'),

test\_preparation\_course=request.form.get('test\_preparation\_course'),

reading\_score=float(request.form.get('writing\_score')),

writing\_score=float(request.form.get('reading\_score')))

pred\_df=data.get\_data\_as\_data\_frame()

print(pred\_df)

predict\_pipeline=PredictPipeline()

results=predict\_pipeline.predict(pred\_df)

return render\_template('home.html',results=results[0])

if \_\_name\_\_=="\_\_main\_\_":

# app.run(host="0.0.0.0",port=8080)

app.run(host='0.0.0.0', port=8080)

**IN THE requirements.txt:**

pandas

numpy

seaborn

matplotlib

scikit-learn

catboost

xgboost

dill

Flask

**IN THE setup.py:**

from setuptools import find\_packages,setup

from typing import List

HYPEN\_E\_DOT='-e .'

def get\_requirements(file\_path:str)-&gt;List[str]:

'''

this function will return the list of requirements

'''

requirements=[]

with open(file\_path) as file\_obj:

requirements=file\_obj.readlines()

requirements=[req.replace("\n","") for req in requirements]

if HYPEN\_E\_DOT in requirements:

requirements.remove(HYPEN\_E\_DOT)

return requirements

setup(

name='mlproject',

version='0.0.1',

author='Krish',

author\_email='krishnaik06@gmail.com',

packages=find\_packages(),

install\_requires=get\_requirements('requirements.txt')

)

<!-- image -->

IN THE NOTEBOOK FOLDER ITSELF WE HAVING THE

SO MY PROBLEM IS I FORGETTEN THE THINGS ABOUT THIS PROJECT KINDLY HELP ME THAT EXPLAIN EVERYTHING ABOUT THE PROJECT LIKE
WORK FLOW FROM THE BEGINNING LIKE CREATING THE ENVIRONMENT AFTER THAT WE ARE GOING TO SETUP THE THINGS
EXPLAIN HOW THESE STRUCTURE IS MAINTAINED  
AFTER THIS EXPLAIN ME THE LOGCI OF THE CODE THAT I HAVE POSTED AND HOW THEY ARE RELATED EVERY SINGLE FILE
LIKE FIRST WE DONE MODEL IMPLEMENTATION THEN CREATED THE COMPONENTS FOLDER AND DONE THE DATA INGESTION, DATA TRANSFORMATION AND MODEL TRAINING WHILE DOING THIS THINGS WE ARE CREATED THE ARTIFACTS FOLDER AND ALSO WHILE DOING THIS WE ARE UPDATING EVERY THING IN THE TEMPLATES FOLDER THEN WE ARE DOING THE PIPELINE PART EXCEPTION, LOGGER AND UTILS 

AFTER THAT WE ARE USING THE TEMPLATES FOLDER FOR CREATING THE APP USING THE FLASK ALSO USED THE DOCKR FILE **MY QUESTION ARE:**

1 – EXPLAIN THE TOTAL WORKFLOW OF THIS PROJECT LIKE AFTER THIS PART WE HAVE TO DO THIS PART LIKE THAT 
2 – EXPLAIN DATA INGESTION, DATA TRANSFORMATION, MODEL TRAINING THE WHOLE CODE LOGIC OF THE CODE AFTER IMPLEMENTING HOW WE ARE CREATING THE ARTIFACTS FOLDER 
3 – EXPLAIN ABOUT THE PIPELINE STRUCTURE AND HOW WE ARE COMBINING THESE THINGS IAM NOT GETTING THAT AFTER CREATING AND IMPLEMENTING HOW THER ARE CONNECTED EXPLAIN THE WORKFLOW DETAILED MANNER AS PER THE CODES AND PICTURE THAT I SHARED
4 – WHICH ALGORITHM PERFORMED WELL ?
5 – ALSO EXLAIN THE TEMPLATES FOLDER THAT I HAVE MENTIONED ABOVE 
6 – HOW THESE ARE IMPLMENTED INTO THE PIPELINE, EXPLAIN WHAT IS THE PIPELINE AND ALSO EXPLAIN HOW WE ARE DOING THESE WITH THE VISUALIZATION THING 
7 – DO EVERYTHING IN VISUALIZING WAY SO THAT I CAN UNDERSTAND MORE
8 -  EXPLAIN THE APP.PY PART HOW THE FLASK IS GOING TO BUILD ALL THESE THINGS SO ATLAST THE FLASK GONNA IMPLEMENT THE WEBAPP SO HOW THESE WORKFLOW IS DONE I NEED CLREA AND DETAILED EXPLANATION OF THAT WORKFLOW 
9 – HOW THEY ARE MAPPING THE THINGS FROM ONE FOLDER TO ANOTHER SHOW THE VISUALIZATION WAY 
10 – HOW THESE THINGS ARE DEPLOYED INTO AWS USING THE CI/CD PIPELINE

11 – EXPLAIN WHAT IS CI/CD PIPELINE AND HOW THESE ARE WORKING IN THESE PROJECT WHAT HAPPEN IF IT IS NOT PRESENTED

EXPLAIN ME IN  A CRYSTAL CLEAR WAY THAT EVEN THE 5 YEAR OLD BOY CAN UNDERSTAND THIS PROJECT AIM, WORLFLOW AND PROBLEM STATEMENT