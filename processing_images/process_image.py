import numpy as np
import cv2

#------------------------Function to display image using opencv------------------------
def show_image(image,image_title: str = ""):
    # import opencv
    import cv2
    
    cv2.imshow(image_title,image) # Display image
    cv2.waitKey(0) # Wait for key to be pressed(any key)
    cv2.destroyAllWindows() # close the program
    
    
def preprocess(image,
               save_image = False,
               dilate_image = False,
               filepath : str = "",
               filename : str = "",
               extension : str = ""
              ):
    
    import cv2
    # Read image using open-cv
    image = cv2.imread(image)
    # Make BGR image to GRAY Scale image
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if not save_image:
        cv2.imwrite(filepath + "/" + filename + extension,image)
        
    # make image blur
    blur_image = cv2.GaussianBlur(gray_image,(9,9),0)
    
    #
    threshold_image = cv2.adaptiveThreshold(blur,
                                            255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            11,
                                            2
                                           )
    # Invert the colors of image because we need to find grid edges.
    output_image = cv2.bitwise_not(threshold_image,threshold_image)
    
    # dilate image
    if not dilate_image:
        
        kernel = [[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]]
        kernel = np.array(kernel)
        
        output_image = cv2.dilate(output_image,kernel)
        
    return output_image


def plot_contours(image):
    contours,hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    
    drawContours = cv2.drawContours(image,contours,-1,(226,3,255),3)
    
    return drawContours,contours




