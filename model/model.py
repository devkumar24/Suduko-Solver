from keras.models import Sequential
from keras.layers import Conv2D,Dropout,Flatten,Dense,MaxPool2D
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping,ModelCheckpoint



#------------------------------------Create Model-----------------------------------------
def create_model():
    # Create CNN_model
    model = Sequential()

    # Add layers to models
    # layer conv2d_1
    model.add( Conv2D( filters = 64, kernel_size = (3,3), activation = "relu", input_shape = (28,28,1) ) )
    # layer conv2d_2
    model.add( Conv2D( filters = 128, kernel_size = (3,3), activation = "relu") )
    model.add( MaxPool2D(pool_size = (2,2)))
    # layer conv2d_3
    model.add( Conv2D( filters = 128, kernel_size = (3,3), activation = "relu") )
    model.add( MaxPool2D(pool_size = (2,2)))
    model.add( Flatten())
    model.add( Dense( units = 256,activation = "relu"))
    model.add( Dense( units = 11,activation = "softmax"))
    # create summary of model
    model.summary()

    # compile model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        
    return model
#-------------------training with/without KFold cross validation--------------------------
def train_model(X,y,filepath,filename,k_fold = False,n_fold = 3,split = 0.15,save_model = True):
    """
    Docstring: It takes an input X,y , X-> training set(it consist of images)
               and y-> label.
               
                Args: 
                    k_fold - It tells whether we have to use KFold Cross validation or not.
                    
                        For Reference for KFold: ###############
                    n_fold - It tells the value of k, that is refer to k_fold.
                    
                    split - It refers to the splitting of X and y data.
                    
                    filepath - Where we want to store our model.
                    
                    filename - What will be the name of our model.
                    
                    save_model - Whether we want to store our model or not.
                    
    
    Syntax: 
        Output refers to 3 args i.e., k_fold,n_fold and split.
            1. k_fold and n_fold
            2. split
        1. 
            history,accuracy,loss = train_model(X,y,True,5)
            
            It is having k_fold = True and make 5 fold cross valodation, means make 5 sets of data.
            
        2.
            history,accuracy,loss = train_model(X,y,0.15)
            
            It is having split = 0.15, it means that it will split the training data into 0.85:0.15 ratio
            training_set:test_set.
            
    
    Return: 
        It will return 3 list history(which contains training_loss,
                                                     training_accuracy,
                                                     test_loss,
                                                     test_accuracy)
                            ,accuracy and loss.
                            
        It will be save our model, if we wish to, But it is recommended to store model, for usage of other
        functions like predict function.
    """
    
    # make list of outputs which we want to return.
    accuracy = list()
    loss = list()
    history = list()
    
    # if k_fold = True
    if not k_fold:
        #prepare n_fold cross validation
        kfold = KFold(n_splits=n_fold,shuffle=True,random_state=42)
        
        for x_train,x_test in kfold.split(X):
            # create model
            model = create_model()
            # select rows for train and test
            train_X,train_y,test_X,test_y = X[x_train],y[x_train],X[x_test],y[x_test]
            # make checkpoints as callbacks
            checkpoint = ModelCheckpoint(filepath = "./digit_classifier.h5",
                                         monitor = "accuracy",
                                         verbose = 1,
                                         save_best_only=True,
                                         mode = "auto"
                                        )
            # make earlystopping as callbacks
            earlyStopping = EarlyStopping(monitor = "accuracy",
                                          min_delta = 0.005,
                                          patience = 5,
                                          verbose = 1,
                                          mode = "auto",
                                          restore_best_weights = True
                                         )
            
            # training of model
            hist = model.fit(x = train_X,
                             y = train_y,
                             batch_size = 128,
                             epochs = 20,
                             validation_data = (test_X,test_y),
                             callbacks = [checkpoint,earlyStopping]
                            )
            # evaulation model(calculate loss and accuracy)
            testing_loss,testing_acc = model.evaluate(test_X,test_y)
            
            # for saving our model
            if save_model:
                model.save(filepath + "/" + filename + ".h5")
                
                print(f"Model saved to {filepath} with {filename}.h5.")
            
            history.append(hist.history)
            loss.append(testing_loss)
            accuracy.append(testing_acc)
            # return all list history accuarcy and loss
            return history,accuracy,loss
        
    
    # if k_fold = False
    else:
        
        # total examples in dataset
        total_samples = X.shape[0]
        
        # train samples in dataset on basis on split
        train_samples = total_samples * (1 - split)
        
        # testing samples in dataset on basis on split
        test_samples = total_samples - train_samples
        
        # make training and testing data 
        (train_X,train_y,test_X,test_y) = ( X[:train_samples],
                                            y[:train_samples],
                                            X[train_samples:test_samples],
                                            y[train_samples:test_samples]
                                        )
        
        # create model
        model = create_model()
        
        # make checkpoints as callbacks
        checkpoint = ModelCheckpoint( filepath = "./digit_classifier.h5",
                                      monitor = "accuracy",
                                      verbose = 1,
                                      save_best_only=True,
                                      mode = "auto"
                                )
        
        # make earlystopping as callbacks
        earlyStopping = EarlyStopping( monitor = "accuracy",
                                       min_delta = 0.07,
                                       patience = 5,
                                       verbose = 1,
                                       mode = "auto",
                                       restore_best_weights = True
                                )
            
        # training of model  
        hist = model.fit( x = train_X,
                          y = train_y,
                          batch_size = 128,
                          epochs = 20,
                          validation_data = (test_X,test_y),
                          callbacks = [checkpoint,earlyStopping]
                    )
        
        # for saving our model
        if save_model:
            model.save(filepath + "/" + filename + ".h5")
                
            print(f"Model saved to {filepath} with {filename}.h5.")
        
        # evaulation model(calculate loss and accuracy)
        testing_loss,testing_acc = model.evaluate(test_X,test_y)
            
        history.append(hist.history)
        loss.append(testing_loss)
        accuracy.append(testing_acc)
        
        # return all list history accuarcy and loss
        return history,accuracy,loss    
#------------------------------This function create plot---------------------------------- 
def plot(history,save_plots = False):
    # as history contains many inside history, so we iterate it, if only len(history) > 1, because of k_fold.
    if len(history) > 1:
        
        for i in range(len(history)):

            loss = history[i]['loss']
            accuracy = history[i]['accuracy']
            val_loss = history[i]['val_loss']
            val_accuracy = history[i]['val_accuracy']

            # plot loss
            plt.subplot(2,1,1)
            plt.title("LOSS GRAPH")
            plt.plot(loss,c = "red",label = "TRAIN_LOSS")
            plt.plot(val_loss,c = "green",label = "VAL_LOSS")
            plt.xlabel("EPOCHS")
            plt.ylabel("LOSS")
            plt.legend()

            #plot accuracy
            plt.subplot(2,1,2)
            plt.title("ACCURACY GRAPH")
            plt.plot(accuracy,c = "red",label = "TRAIN_ACCURACY")
            plt.plot(val_loss,c = "green",label = "VAL_ACCURACY")
            plt.xlabel("EPOCHS")
            plt.ylabel("ACCURACY")
            plt.legend()
            plt.show()
            
            if not save_plots:
                plt.savefig("./evaluation/LossAndAccuracyPlot.png")
    
    else: 
            loss = history['loss']
            accuracy = history['accuracy']
            val_loss = history['val_loss']
            val_accuracy = history['val_accuracy']

            # plot loss
            plt.subplot(2,1,1)
            plt.title("LOSS GRAPH")
            plt.plot(loss,c = "red",label = "TRAIN_LOSS")
            plt.plot(val_loss,c = "green",label = "VAL_LOSS")
            plt.xlabel("EPOCHS")
            plt.ylabel("LOSS")
            plt.legend()

            #plot accuracy
            plt.subplot(2,1,2)
            plt.title("ACCURACY GRAPH")
            plt.plot(accuracy,c = "red",label = "TRAIN_ACCURACY")
            plt.plot(val_loss,c = "green",label = "VAL_ACCURACY")
            plt.xlabel("EPOCHS")
            plt.ylabel("ACCURACY")
            plt.legend()
            plt.show()
            
            if not save_plots:
                plt.savefig("./evaluation/LossAndAccuracyPlot.png")
#----------------------------------Prediction function------------------------------------
def predict(model_name_with_extension,test_image,plot_given_image = False):
    # load model for prediction
    model = load_model(model_name_with_extension)
    
    # pre process image before giving to model
    pre_process_image = preprocess(test_image)
    
    # make predictions of image
    predictions = model.predict_classes(pre_process_image,verbose = 0)
    
    
    predictions = predictions[0]
    # print predicted digit
    print(f"Predicted Value--->{predictions}.")
    
    if not plot_given_image:
        # plot given image
        plot_image(test_image)
    
    else:
        return None
    
#-----------------------------------------------------------------------------------------