# EmotionDetection

## Installation


```bash
pip install numpy
pip install keras
pip install tensorflow
pip install pandas
pip install matplotlib
pip install seaborn
```

## Import Libraries
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```


## Dataset Overview
You need to make sure you know the location of the file in directory
```python
file_path = '/Users/shirzlotnik/emotion_dataset/fer2013.csv' # file path in the computer

data = pd.read_csv(file_path)
# check data shape
print(data.shape)
# preview first 5 row of data
print(data.head(5))
```
