import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import deeplabv3_plus
from metrics import dice_loss, dice_coef, iou



"""Global parameters"""

H = 512
W = 512

#creating directory if not exists

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x,y):
    x,y = shuffle(x,y,random_state=42)
    return x,y

def load_data(path):
    x = sorted(glob(os.path.join(path,"image","*png")))
    y = sorted(glob(os.path.join(path,"mask","*png")))
    return x,y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    x, y = tf.numpy_function(_parse,[x,y],[tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X,Y,batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset
    

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files"""
    create_dir("files")

    """ Hyperparameters """
    batch_size=1
    lr=1e-4
    num_epochs=1
    model_path = os.path.join("files" , "model.h5")
    csv_path = os.path.join("files" , "data.csv")

    # Print the values of model_path and csv_path
    print("Before joining paths:")
    print("model_path:", model_path)
    print("csv_path:", csv_path)

    # Perform any other operations as needed

    # Now, join the paths
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    # Print the values again to verify
    print("After joining paths:")
    print("model_path:", model_path)
    print("csv_path:", csv_path)

    # Continue with the rest of your code


    """ Dataset """
    dataset_path = "new"
    train_path = os.path.join(dataset_path,"train")
    valid_path = os.path.join(dataset_path,"test")

    x_train,y_train = load_data(train_path)
    x_train,y_valid = shuffling(x_train,y_train) 
    x_valid,y_valid = load_data(valid_path)

    print("train size",len(x_train))
    print("train size",len(y_train))
    print("valid size",len(x_valid))
    print("valid size",len(y_valid))


    train_dataset = tf_dataset(x_train, y_train, batch = batch_size)
    valid_dataset = tf_dataset(x_valid, y_valid, batch = batch_size)

    # for x,y in train_dataset:
    #     print(x.shape,y.shape)
    #     break

    """ Model """
    model = deeplabv3_plus((H, W, 3))
    model.compile(loss=dice_loss,optimizer=Adam(lr) ,metrics=[dice_coef, iou, Recall(),Precision()])
    # model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_weights_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
