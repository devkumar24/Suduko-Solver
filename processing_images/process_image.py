

# Import packages
import cv2
import numpy as np

# import other files
from processing_images import helper


#-------------------------------------------------------------------------------------
def pre_process_images(image_path):
    """
    Docstring:
        This is function which return the all the 81 grids, from suduko puzzle image.
        First it will preprocess Image, and then make crop the puzzle image.
        Secondly, it will get the points of all the rectangle grid, and at last
        it will make the 81 grids, i.e., we will get our required 81 images for
        suudko puzzle.

    Args:
        image_path -> Path of the image or URL of the image of which we want to make
                      our grids.

    Return:
        It will return our 81 grids, of suduko puzzle.
    """
    image = helper.preprocess(image_path)
    crop = helper.crop_image(image)
    boxes = helper.grid_box(crop)
    digit = helper.get_digit(crop,boxes,28)
    return digit
#--------------------------------------------------------------------------------
def showImage(image_path,show_img = False):
    """
    Function to show preprocess image.
    """
    image = helper.preprocess(image_path)
    if show_img:
        helper.show_image(image)
    
    return image
#--------------------------------------------------------------------------------
def show_cropImage(image_path,show_img = False):
    """
    Function to show cropped image
    """
    image = showImage(image_path)
    crop = helper.crop_image(image)
    
    if show_img:
        helper.show_image(crop)
        
    return crop