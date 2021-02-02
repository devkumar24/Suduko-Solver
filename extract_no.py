

# import packages
import numpy as np
import cv2


# import load_model package for prediction
from keras.models import load_model


# import other files
from processing_images import helper
from processing_images import process_image

#--------------------------------------------------------------------------------------------------
def predict(image_grid,model_name = None):
    """
    Docstring:
        This function will make the prediction of the rectangle grid, which is given as input.
        **ACCURACY of model is 0.99.

    Args:
        image_grid -> The grid of which we want to predict whether its a number or blank image.
        model_name -> It is in string format, which is basically the path of MNIST model is stored.

    Return:
        It will return the prediction of the rectangle grid
        Label: 1 to 9 and "."

    ** There might be case sometimes 0 occur and wrong number is predicted, it means that there is 
       noise in image or cropped image is incorrectly cropped.
    """
    model = load_model(model_name)
    
    image = np.reshape(image_grid,(1,28,28,1))
    
    if image.sum() > 4000:
        prediction = model.predict_classes(image, verbose = 0)
        prediction = prediction[0]
        return prediction
    else:
        return "."
#--------------------------------------------------------------------------------------------------
def extract_number(digit,model_name = None):
    """
    Docstring:
        This is the most important function because this function prediction of grid, and without 
        the extraction of number, we can't create the suduko solver.
    Args:
        digits -> Images of 81 grids, which is basically a list of images.
        model_name -> It is in string format, which is basically the path of MNIST model is stored.

    Return:
        It will return the numpy.ndarray of prediction of numbers and blanks, which is basically 
        a grid of 9*9 which is actually a suduko grid, which is used to solve suduko puzzle.
    """
    grid = []
    for i in range(len(digit)):
        d = digit[i]
        _,mask = cv2.threshold(d,128,255,cv2.THRESH_BINARY)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.dilate(mask,element)
        mask = cv2.erode(mask,element, iterations = 1)
        
        pred = predict(mask,model_name)
        grid.append(pred)
    
    grid = np.reshape(grid,(9,9))
    grid = grid.T
    return grid
#--------------------------------------------------------------------------------------------------
def save_grid(grid,filename = None,save = True):
    """
    Docstring:
        This will save the grids for future use, or we can use the grids to create our suduko dataset.
        It will create a file called puzzles.txt and append the predicted digits and blank to file, which
        is suduko grid.

    Return:
        It will return the numpy.ndarray of prediction of numbers and blanks, which is basically 
        a grid of 9*9 which is actually a suduko grid, which is used to solve suduko puzzle.
    """
    if save:
        file = open(filename,"a")
        file.write("\n") 
        if type(grid) == list:
            for i in range(len(grid)):
                file.write(str(grid[i]))
        else:
            if type(grid) == np.ndarray:
                grid = list(grid)
                for i in range(len(grid)):
                    for j in grid[i]:
                        file.write(str(j))
         
        file.close()
    return np.array(grid)
#--------------------------------------------------------------------------------------------------