
# Import Packages
import numpy as np
import keras
import tensorflow as tf
import pandas as pd

#-----------------------------------------------------
def ydata(filename : str = ""):
    try:
        import keras
    except ModuleNotFoundError:
        print("Module Not found!!!")
    
    
    train_data = pd.read_csv(filename)
    labels = train_data.label
    y = labels.values
    y = keras.utils.to_categorical(y)
    
    return y
#-----------------------------------------------------
def Xdata(filename : str = ""):
    X_data = pd.read_csv(filename)
    for i in X_data.columns:
        
        if i == "label":
            X_data = X_data.drop("label",axis = 1)

        else: 
            raise KeyError
        
        X = X_data.values
        X = X/255.
        X = X.reshape((-1,28,28,1))
        X = X/255.
        X = X.reshape((-1,28,28,1))

        return X

