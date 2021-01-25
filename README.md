# EmotionDetection

```python

```

## Install Dataset
install dataset from this [kaggle](https://www.kaggle.com/lxyuan0420/facial-expression-recognition-using-cnn/data) project   
fer2013.csv


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

Than it will print this if everything work fine  
Using TensorFlow backend.  

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
(35887, 3)

Index | emotion | pixels | Usage
------------ | ------------- | ------------- | -------------
0 | 0 | 0 80 82 72 58 58 60 63 54 58 60 48 ... | Training
1 | 0 | 151 150 147 155 148 133 111 140 170... | Training
2 | 2 | 231 212 156 164 174 138 161 173 182... | Training
3 | 4 | 24 32 36 30 32 23 19 20 30 41 21 22... | Training
4 | 6 | 4 0 0 0 0 0 0 0 0 0 0 0 3 15 23 28 ... | Training


print how many images there is for each usage
```python
# check usage values
# 80% training, 10% validation and 10% test
print(data.Usage.value_counts())
```
Training       28709  
PublicTest      3589  
PrivateTest     3589  
Name: Usage, dtype: int64  


```python
 # check target labels
emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
print(emotion_counts)
```
 A | emotion | number  
------------ | ------------- | ------------- 
0 | Angry | 4953  
1 | Digust | 547  
2 | Fear | 5121  
3 | Happy | 8989  
4 | Sad | 6077  
5 | Surprise | 4002
6 | Neutral | 6198

```python
# Plotting a bar graph of the class distributions
plt.figure(figsize=(6,4))
sns.barplot(emotion_counts.emotion, emotion_counts.number)
plt.title('Class distribution')
plt.ylabel('Number', fontsize=12)
plt.xlabel('Emotions', fontsize=12)
plt.show()
```
![class distribution](https://github.com/shirzlotnik/EmotionDetection/blob/main/class_distribution.png?raw=true)



## plot some images

```python
def row2image(row):
    pixels, emotion = row['pixels'], emotion_map[row['emotion']]
    img = np.array(pixels.split())
    img = img.reshape(48,48)
    image = np.zeros((48,48,3))
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return np.array([image.astype(np.uint8), emotion])

plt.figure(0, figsize=(12,6))
for i in range(1,8):
    face = data[data['emotion'] == i-1].iloc[0]
    img = row2image(face)
    plt.subplot(2,4,i)
    plt.imshow(img[0])
    plt.title(img[1])

plt.show()
```
![Images](https://github.com/shirzlotnik/EmotionDetection/blob/main/angry.png?raw=true)


## Pre-processing data

1. Splitting dataset into 3 parts: train, validation, test
2. Convert strings to lists of integers
3. Reshape to 48x48 and normalise grayscale image with 255.0
4. Perform one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]

```python

#split data into training, validation and test set
data_train = data[data['Usage']=='Training'].copy()
data_val   = data[data['Usage']=='PublicTest'].copy()
data_test  = data[data['Usage']=='PrivateTest'].copy()
print("train shape: {}, \nvalidation shape: {}, \ntest shape: {}".format(
        data_train.shape, data_val.shape, data_test.shape))
```

train shape: (28709, 3),   
validation shape: (3589, 3),  
test shape: (3589, 3)  

```python
# barplot class distribution of train, val and test
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

"""
# OLD VERSION WITH NUMVERS ABOVE THE BARS
def setup_axe(axe,df,title):
    df['emotion'].value_counts(sort=False).plot(ax=axe, kind='bar', rot=0)
    axe.set_xticklabels(emotion_labels)
    axe.set_xlabel("Emotions")
    axe.set_ylabel("Number")
    axe.set_title(title)
    
    # set individual bar lables using above list
    for i in axe.patches:
        # get_x pulls left or right; get_height pushes up or down
        axe.text(i.get_x()-.05, i.get_height()+120, \
                str(round((i.get_height()), 2)), fontsize=14, color='dimgrey',
                    rotation=0)


   
fig, axes = plt.subplots(1,3, figsize=(20,8), sharey=True)
setup_axe(axes[0],data_train,'train')
setup_axe(axes[1],data_val,'validation')
setup_axe(axes[2],data_test,'test')
plt.show()
"""
```

### OLD VERSION
![Charts](https://github.com/shirzlotnik/EmotionDetection/blob/main/chart1.png?raw=true)

Notice that the later two subplots share the same y-axis with the first subplot.  
The size of train, validation, test are 80%, 10% and 10%, respectively.  
The exact number of each class of these datasets are written on top of their x-axis bar.  

```python
# current version

def Setup_axe(df,title):
    emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    emotion_Count = df['emotion'].value_counts(sort=False).reset_index()
    emotion_Count.columns = ['emotion', 'number']
    emotion_Count['emotion'] = emotion_Count['emotion'].map(emotion_map)
    
    sns.barplot(emotion_Count.emotion, emotion_Count.number)
    plt.title(title)
    plt.ylabel('Number', fontsize=12)
    plt.xlabel('Emotions', fontsize=12)
    plt.show()
    
    
def count_emotion_in_columns(df):
    """
    df: data
    the function sort the data by usage and then by emotion
    return: sorted data by usage
    """
    df_train = df[df['Usage']=='Training'].copy()
    df_val   = df[df['Usage']=='PublicTest'].copy()
    df_test  = df[df['Usage']=='PrivateTest'].copy()
    
    train1 = df_train['emotion'].value_counts().sort_index()
    val1 = df_val['emotion'].value_counts().sort_index()
    test1 = df_test['emotion'].value_counts().sort_index()
    
    train_sorted = sorted(train1.items(), key = lambda d: d[1], reverse = True)
    val_sorted = sorted(val1.items(), key = lambda d: d[1], reverse = True)
    test_sorted = sorted(test1.items(), key = lambda d: d[1], reverse = True)
    
    return train_sorted, val_sorted, test_sorted



def print_Usage_Information(columns_count,data_sorted, usage, emotion_labels):
    """
    data_sorted: the sorted data by usage and then by emotion from count_emotion_in_columns()
    usage: string, the usage
    columns_count: data_train.shape[0]- how many of that usage total
    print number of *emotion* in *usage*
    """
    for info in data_sorted:
        print('Number of {} in {} = {} => {}%'.format(emotion_labels[info[0]],
              usage, info[1], (info[1]/columns_count)*100))
              
              
trainSort, valSort, testSort = count_emotion_in_columns(data)


Setup_axe(data_train,'train')
print_Usage_Information(data_train.shape[0],trainSort,'training data',emotion_labels)
Setup_axe(data_val,'validation')
print_Usage_Information(data_val.shape[0],valSort,'validation data',emotion_labels)
Setup_axe(data_test,'test')
print_Usage_Information(data_test.shape[0],testSort,'testing data',emotion_labels)

```

### CURRENT VERSION

![Traininh Chart](https://github.com/shirzlotnik/EmotionDetection/blob/main/train1.png?raw=true)  

Number of Happy in training data = 7215 => 25.13149186666202%  
Number of Neutral in training data = 4965 => 17.294228290779895%  
Number of Sad in training data = 4830 => 16.82399247622697%  
Number of Fear in training data = 4097 => 14.270786164617366%  
Number of Angry in training data = 3995 => 13.91549688251071%  
Number of Surprise in training data = 3171 => 11.045316799609878%  
Number of Disgust in training data = 436 => 1.5186875195931588%  

![Validation Chart](https://github.com/shirzlotnik/EmotionDetection/blob/main/val1.png?raw=true)  

Number of Happy in validation data = 895 => 24.93730844246308%  
Number of Sad in validation data = 653 => 18.19448314293675%  
Number of Neutral in validation data = 607 => 16.912789077737532%  
Number of Fear in validation data = 496 => 13.820005572582891%  
Number of Angry in validation data = 467 => 13.011981053218166%  
Number of Surprise in validation data = 415 => 11.56310950125383%  
Number of Disgust in validation data = 56 => 1.560323209807746%  

![Testing Chart](https://github.com/shirzlotnik/EmotionDetection/blob/main/test1.png?raw=true)  

Number of Happy in testing data = 879 => 24.49150181108944%  
Number of Neutral in testing data = 626 => 17.44218445249373%  
Number of Sad in testing data = 594 => 16.550571189746446%  
Number of Fear in testing data = 528 => 14.711618835330176%  
Number of Angry in testing data = 491 => 13.68069100027863%  
Number of Surprise in testing data = 416 => 11.590972415714685%  
Number of Disgust in testing data = 55 => 1.5324602953468933%  
  
```python
#initilize parameters
num_classes = 7 
width, height = 48, 48
num_epochs = 50
batch_size = 64
num_features = 64
```

```python
"""
CRNO stands for Convert, Reshape, Normalize, One-hot encoding
(i) convert strings to lists of integers
(ii) reshape and normalise grayscale image with 255.0
(iii) one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]
"""

def CRNO(df, dataName):
    df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
    data_X = np.array(df['pixels'].tolist(), dtype='float32').reshape(-1,width, height,1)/255.0   
    data_Y = to_categorical(df['emotion'], num_classes)  
    print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))
    return data_X, data_Y

    
train_X, train_Y = CRNO(data_train, "train") #training data
val_X, val_Y     = CRNO(data_val, "val") #validation data
test_X, test_Y   = CRNO(data_test, "test") #test data
```
train _X shape: {},  train _Y shape: (28709, 48, 48, 1)  
val _X shape: {},  val _Y shape: (3589, 48, 48, 1)  
test _X shape: {},  test _Y shape: (3589, 48, 48, 1)  


## Building CNN Model
### CNN Architecture:
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Conv -> BN -> Activation -> Conv -> BN -> Activation -> MaxPooling
    Flatten
    Dense -> BN -> Activation
    Dense -> BN -> Activation
    Dense -> BN -> Activation
    Output layer
    
    
```python
model = Sequential()

#module 1
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), input_shape=(width, height, 1), data_format='channels_last'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#module 2
model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(2*num_features, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#module 3
model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(num_features, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#flatten
model.add(Flatten())

#dense 1
model.add(Dense(2*2*2*num_features))
model.add(BatchNormalization())
model.add(Activation('relu'))

#dense 2
model.add(Dense(2*2*num_features))
model.add(BatchNormalization())
model.add(Activation('relu'))

#dense 3
model.add(Dense(2*num_features))
model.add(BatchNormalization())
model.add(Activation('relu'))

#output layer
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), 
              metrics=['accuracy'])

model.summary()
```

 Layer (type) | Output Shape | Param #  
------------ | ------------- | ------------- 
conv2d_1 (Conv2D)            | (None, 46, 46, 256) | 2560  
batch_normalization_1 (Batch | (None, 46, 46, 256) | 1024  
activation_1 (Activation)    | (None, 46, 46, 256) | 0  
conv2d_2 (Conv2D)            | (None, 46, 46, 256) | 590080  
batch_normalization_2 (Batch | (None, 46, 46, 256) | 1024 
activation_2 (Activation)    | (None, 46, 46, 256) | 0  
max_pooling2d_1 (MaxPooling2 | (None, 23, 23, 256) | 0  
conv2d_3 (Conv2D)            | (None, 23, 23, 128) | 295040  
batch_normalization_3 (Batch | (None, 23, 23, 128) | 512 
activation_3 (Activation)    | (None, 23, 23, 128) | 0   
conv2d_4 (Conv2D)            | (None, 23, 23, 128) | 147584   
batch_normalization_4 (Batch | (None, 23, 23, 128) | 512  
activation_4 (Activation)    | (None, 23, 23, 128) | 0   
max_pooling2d_2 (MaxPooling2 | (None, 11, 11, 128) | 0  
conv2d_5 (Conv2D)            | (None, 11, 11, 64)  | 73792   
batch_normalization_5 (Batch | (None, 11, 11, 64)  | 256  
activation_5 (Activation)    | (None, 11, 11, 64)  | 0   
conv2d_6 (Conv2D)            | (None, 11, 11, 64)  | 36928   
batch_normalization_6 (Batch | (None, 11, 11, 64)  | 256  
activation_6 (Activation)    | (None, 11, 11, 64)  | 0  
max_pooling2d_3 (MaxPooling2 | (None, 5, 5, 64)    | 0    
flatten_1 (Flatten)          | (None, 1600)        | 0   
dense_1 (Dense)              | (None, 512)         | 819712   
batch_normalization_7 (Batch | (None, 512)         | 2048   
activation_7 (Activation)    | (None, 512)         | 0   
dense_2 (Dense)              | (None, 256)         | 131328   
batch_normalization_8 (Batch | (None, 256)         | 1024   
activation_8 (Activation)    | (None, 256)         | 0   
dense_3 (Dense)              | (None, 128)         | 32896  
batch_normalization_9 (Batch | (None, 128)         | 512  
activation_9 (Activation)    | (None, 128)         | 0   
dense_4 (Dense)              | (None, 7)           | 903   

Total params: 2,137,991  
Trainable params: 2,134,407  
Non-trainable params: 3,584  

  
    
## data generator  
 
```python
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)


es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)

history = model.fit_generator(data_generator.flow(train_X, train_Y, batch_size),
                                steps_per_epoch=len(train_X) / batch_size,
                                epochs=num_epochs,
                                verbose=2, 
                                callbacks = [es],
                                validation_data=(val_X, val_Y))
```
Epoch 1/50  
 - 37s - loss: 1.7037 - acc: 0.3242 - val_loss: 1.6681 - val_acc: 0.3589  
Epoch 2/50  
 - 30s - loss: 1.4228 - acc: 0.4470 - val_loss: 1.4414 - val_acc: 0.4450  
Epoch 3/50  
 - 30s - loss: 1.2625 - acc: 0.5140 - val_loss: 1.5380 - val_acc: 0.4606  
Epoch 4/50  
 - 30s - loss: 1.1799 - acc: 0.5468 - val_loss: 1.3059 - val_acc: 0.5102  
Epoch 5/50  
 - 30s - loss: 1.1281 - acc: 0.5658 - val_loss: 1.1986 - val_acc: 0.5366  
Epoch 6/50  
 - 30s - loss: 1.0889 - acc: 0.5892 - val_loss: 1.2567 - val_acc: 0.5308  
Epoch 7/50  
 - 30s - loss: 1.0636 - acc: 0.5970 - val_loss: 1.0932 - val_acc: 0.5913  
Epoch 8/50  
 - 30s - loss: 1.0317 - acc: 0.6090 - val_loss: 1.3399 - val_acc: 0.4912  
Epoch 9/50  
 - 30s - loss: 1.0131 - acc: 0.6192 - val_loss: 1.0130 - val_acc: 0.6222  
Epoch 10/50  
 - 30s - loss: 0.9876 - acc: 0.6263 - val_loss: 1.1529 - val_acc: 0.5857  
Epoch 11/50  
 - 30s - loss: 0.9707 - acc: 0.6336 - val_loss: 1.1352 - val_acc: 0.5756  
Epoch 12/50  
 - 30s - loss: 0.9510 - acc: 0.6380 - val_loss: 1.1439 - val_acc: 0.5612  
Epoch 13/50  
 - 30s - loss: 0.9331 - acc: 0.6493 - val_loss: 1.0174 - val_acc: 0.6266  
Epoch 14/50  
 - 30s - loss: 0.9165 - acc: 0.6531 - val_loss: 1.1069 - val_acc: 0.6007  
Epoch 15/50  
 - 30s - loss: 0.8946 - acc: 0.6609 - val_loss: 1.1158 - val_acc: 0.5965  
Epoch 16/50  
 - 30s - loss: 0.8870 - acc: 0.6627 - val_loss: 1.0189 - val_acc: 0.6294  
Epoch 17/50  
 - 30s - loss: 0.8739 - acc: 0.6713 - val_loss: 1.0039 - val_acc: 0.6330  
Epoch 18/50  
 - 30s - loss: 0.8559 - acc: 0.6781 - val_loss: 1.0986 - val_acc: 0.6016  
Epoch 19/50  
 - 30s - loss: 0.8472 - acc: 0.6804 - val_loss: 1.0635 - val_acc: 0.6071  
Epoch 20/50  
 - 30s - loss: 0.8290 - acc: 0.6886 - val_loss: 1.0814 - val_acc: 0.6035  
Epoch 21/50  
 - 30s - loss: 0.8133 - acc: 0.6933 - val_loss: 0.9858 - val_acc: 0.6500  
Epoch 22/50  
 - 30s - loss: 0.7980 - acc: 0.6965 - val_loss: 1.0199 - val_acc: 0.6372  
Epoch 23/50  
 - 30s - loss: 0.7947 - acc: 0.7022 - val_loss: 0.9690 - val_acc: 0.6542  
Epoch 24/50  
 - 30s - loss: 0.7757 - acc: 0.7080 - val_loss: 0.9849 - val_acc: 0.6492  
Epoch 25/50  
 - 30s - loss: 0.7684 - acc: 0.7102 - val_loss: 0.9916 - val_acc: 0.6534  
Epoch 26/50  
 - 30s - loss: 0.7521 - acc: 0.7159 - val_loss: 1.3015 - val_acc: 0.5678  
Epoch 27/50  
 - 30s - loss: 0.7392 - acc: 0.7226 - val_loss: 1.0193 - val_acc: 0.6336  
Epoch 28/50  
 - 30s - loss: 0.7324 - acc: 0.7272 - val_loss: 1.0696 - val_acc: 0.6403  
Epoch 29/50  
 - 30s - loss: 0.7179 - acc: 0.7304 - val_loss: 1.0273 - val_acc: 0.6411  
Epoch 30/50  
 - 30s - loss: 0.7060 - acc: 0.7374 - val_loss: 1.0358 - val_acc: 0.6456  
Epoch 31/50  
 - 30s - loss: 0.6927 - acc: 0.7408 - val_loss: 1.0999 - val_acc: 0.6372  
Epoch 32/50  
 - 30s - loss: 0.6853 - acc: 0.7427 - val_loss: 1.0485 - val_acc: 0.6319  
Epoch 33/50  
 - 30s - loss: 0.6645 - acc: 0.7518 - val_loss: 1.0801 - val_acc: 0.6319  
 
 
 
  # NOT IN SCRIPT
  
 ## Visualize Training Performance
 
 ```python
 fig, axes = plt.subplots(1,2, figsize=(18, 6))
# Plot training & validation accuracy values
axes[0].plot(history.history['acc'])
axes[0].plot(history.history['val_acc'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'], loc='upper left')
plt.show()

```

![Results](https://github.com/shirzlotnik/EmotionDetection/blob/main/graphs2.png?raw=true)

## Evaluate Test Performance

```python
test_true = np.argmax(test_Y, axis=1)
test_pred = np.argmax(model.predict(test_X), axis=1)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(test_true, test_pred)))
```
CNN Model Accuracy on test set: 0.6662  

## More Analysis using Confusion Matrix

```python
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
```
```python
# Plot normalized confusion matrix
plot_confusion_matrix(test_true, test_pred, classes=emotion_labels, normalize=True, title='Normalized confusion matrix')
plt.show()
```

![Normlized_Confusion](https://github.com/shirzlotnik/EmotionDetection/blob/main/normalized_confusion.png?raw=true)


# Future Work
```
(i) To further fine tuning model using grid_search, specifically:
    a. Different optimizer such as Adam, RMSprop, Adagrad.
    b. experimenting dropout with batch-normalization.
    c. experimenting different dropout rates. 

(ii) To collect more data and train the model with balance dataset.
```
```python

```

