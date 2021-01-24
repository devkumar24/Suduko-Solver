import numpy as np
import cv2

#-------------------------------------------------------------------------------------------------------
def create_image(height,width,grayScale = False,grayChannel = False, whiteImage = True):
    """
    This will create white or a black image as per requirement.
    Args:
        height      -> Tells about the height of image. Let say height = 512(pixels).
        width       -> Tells about the width of image. Let say width = 512(pixels).
        grayScale   -> Whether we want grayScale image or not.
        grayChannel -> Whether we want channel or not.
        whiteImage  -> Whther we want white image/ black image.
        
        
    Return:
        It will return an numpy.ndarray of shape (height,width,_), which is basically denotes an image.
    """   
    if whiteImage: # Check whether user want whiteImage or not.
        
        if grayScale: #GrayScale or not
            if grayChannel: #grayChannel or not
                return 255. * np.ones((height,width,1),np.uint8) # return an numpy.ndarray
            else:
                return 255. * np.ones((height,width),np.uint8) # return an numpy.ndarray
        else:
            return 255. * np.ones((height,width,3),np.uint8) # return an numpy.ndarray
        
    else:
        if grayScale:
            if grayChannel:
                return np.zeros((height,width,1),np.uint8)
            else:
                return np.zeros((height,width),np.uint8)
        else:
            return np.zeros((height,width,3),np.uint8)

#-------------------------------------------------------------------------------------------------------        
def show_image(image,image_title: str = ""):
    # import opencv
    import cv2
    
    cv2.imshow(image_title,image) # Display image
    cv2.waitKey(0) # Wait for key to be pressed(any key)
    cv2.destroyAllWindows() # close the program
    
#-------------------------------------------------------------------------------------------------------  
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
    if save_image:
        cv2.imwrite(filepath + "/" + filename + extension,image)
        
    # make image blur
    blur_image = cv2.GaussianBlur(gray_image,(9,9),0)
    
    #
    threshold_image = cv2.adaptiveThreshold(blur_image,
                                            255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            15,
                                            2
                                           )
    # Invert the colors of image because we need to find grid edges.
    output_image = cv2.bitwise_not(threshold_image,threshold_image)
    
    # dilate image
    if dilate_image:
        
        kernel = [[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]]
        kernel = np.array(kernel)
        
        output_image = cv2.dilate(output_image,kernel)
        
    return output_image

#----------------------------------------------------------------------------------------------------
def GRAY_RGB_GRAY(image,GRAY_RGB = True):    
    if GRAY_RGB and (len(image.shape) == 2 or image.shape[2] == 1):
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        return image
    else:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image
    
#------------------------------------------------------------------------------------------------------
def findContours(image):
    """
    Contours are defined as the line joining all the points along the boundary of an 
    image that are having the same intensity. Contours come handy in shape analysis, 
    finding the size of the object of interest, and object detection.
    """
    contours,hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours
#-------------------------------------------------------------------------------------------------------
def plotContours(image,showImage = False):
    
    contours = findContours(image)
    image = GRAY_RGB_GRAY(image)
    drawContours = cv2.drawContours(image,contours,-1,(226,3,255),3)
    
    if showImage:
        show_image(drawContours)
        
    return drawContours

#--------------------------------------------------------------------------------------------------------
def cal_corners(image):
    import operator
    
    contours = findContours(image)
    
    contours = sorted(contours,key = cv2.contourArea,reverse=True)
    
    for_largest = contours[0]
    
    bottom_r, _ = max( enumerate([ coordinate[0][0] + coordinate[0][1] for coordinate in for_largest]), key=operator.itemgetter(1) )
    top_l, _ = min( enumerate([ coordinate[0][0] + coordinate[0][1] for coordinate in for_largest]), key=operator.itemgetter(1) )
    bottom_l, _ = min( enumerate([ coordinate[0][0] - coordinate[0][1] for coordinate in for_largest]), key=operator.itemgetter(1) )
    top_r, _ = max( enumerate([ coordinate[0][0] - coordinate[0][1] for coordinate in for_largest]), key=operator.itemgetter(1) )
    
    return [ for_largest[top_l][0],for_largest[top_r][0],for_largest[bottom_r][0],for_largest[bottom_l][0] ]

#-----------------------------------------------------------------------------------------------------------
def display_points(image,radius = 10, showImage=False):
    corners = cal_corners(image)
    image = GRAY_RGB_GRAY(image)
    points = list()
    for point in corners:
        center = tuple(int(x) for x in point)
        image = cv2.circle(image,center,radius,(226,0,255), -1)
        points.append(center)

    if showImage:
        show_image(image)
    return image,points

#-----------------------------------------------------------------------------------------------------------
def find_largest_polygon(image,threshold = 0.1,n = 4):
    contours = findContours(image)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)
    
    largest_contour = contours[0]
    if n>0:
        
        for contour in largest_contour:
            
            perimeter = cv2.arcLength(contour, True)
            sides = cv2.approxPolyDP(contour,threshold * perimeter,True)
            if len(sides) == n:
                return contour
            
    else:
        return contours[0]
    
#-----------------------------------------------------------------------------------------------------------
def distance(p1,p2):
    return (np.sum((p1-p2)**2)**0.5)

#-------------------------------------------------------------
def crop(image):
    corners = cal_corners(image)
    
    top_l,top_r,bottom_r,bottom_l = corners
    
    side = max([distance(top_l,top_r),distance(bottom_l,bottom_r),distance(top_l,bottom_l),distance(top_r,bottom_r)])
    
    src = np.array([top_l,top_r,bottom_r,bottom_l],dtype='float32')
    dst = [[0,0], [side - 1,0], [side - 1,side - 1], [0,side - 1]]
    dst = np.array(dst,dtype = 'float32')
    m = cv2.getPerspectiveTransform(src,dst)
    get = cv2.warpPerspective(image,m,(int(side),int(side)))
    return get


#-----------------------------------------------------------------------
def grid_box(image):
    shape = image.shape[0]
    size = int(shape/9)
    
    boxes = list()
    for i in range(9):
        for j in range(9):
            p1 = (i*size,j*size)
            p2 = ((i+1)*size,(j+1)*size)
            boxes.append((p1,p2))
    return boxes # return top_left and bottom_right co-ordinates
#----------------------------------------------------------------------
def display_grid_box(image,radius = 5,color = (226,0,255),showImage = False):
    image = crop(image)
    points = grid_box(image)
    
    image = GRAY_RGB_GRAY(image)
    
    for pt in points:
        image = cv2.circle(image,pt[0],radius,color, -1)
        image = cv2.circle(image,pt[1],radius,color,-1)
        
    if showImage:
        show_image(image)
    return image
#----------------------------------------------------------------
def crop_boxes(image,boxes):
    return image[boxes[0][1]:boxes[1][1], boxes[0][0]:boxes[1][0]]
#-------------------------------------------------------------------------------




