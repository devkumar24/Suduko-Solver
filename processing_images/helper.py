# Import packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import operator

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

    cv2.imshow(image_title,image) # Display image
    cv2.waitKey(0) # Wait for key to be pressed(any key)
    cv2.destroyAllWindows() # close the program
#-------------------------------------------------------------------------------------------------------  
def preprocess(image_path = None,save_image = False,dilate_image = False,filename : str = ""):
	"""
	Docstring:
		It will read an image from the given path, and return an Image which is 
		preprocessed by various methods.
		Args:
			1. image_path -> Path of the image
			2. save_image -> Want to save to image, in case anyone copy image_path as URL's.
			3. dilate_image -> If we want to make our image dilate, dilation of image means that 
							   if particular pixel is mapped with a kernel, if result is 1, that
							   means pixel is white and if result is 0, that means pixel is black.
			4. filename -> It is string format, filename is the path + name_of_file + extension 
						   (in which format user want to save image. **NOTE: .jpg is recommended). 
						   It will save the image at that path.

		Return:
			It will return the image.

		Example:
			image_path = "../images/before/puzzle.jpg"
			pre_process_image = preprocess(image_path)

			**We have skip dilate_image arguement, because in the function we already bitwise_not
			  and adaptiveThreshold like methods,so dilate_image, will not effect that much. 
			  If we do dilate_image, then also there is no problem. 
	"""
    # Read image using open-cv
    image = cv2.imread(image_path)
    # Make BGR image to GRAY Scale image
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if save_image:
        cv2.imwrite(filename,image)
        
    # make image blur
    blur_image = cv2.GaussianBlur(gray_image,(9,9),0)
    
    #
    threshold_image = cv2.adaptiveThreshold(blur_image,
                                            255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            11,
                                            2
                                           )
    # Invert the colors of image because we need to find grid edges.
    output_image = cv2.bitwise_not(threshold_image,threshold_image)
    
    # dilate image
    if dilate_image:
        
        kernel = [[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]]
        kernel = np.array(kernel)
        
        output_image = cv2.dilate(output_image,kernel,5)
        
    return output_image
#-------------------------------------------------------------------------------------------------------
def GRAY_RGB_GRAY(image,GRAY_RGB = True):    
	"""
	This function will read a pre_processed image, and check the shape of image, 
	if image has 2 channels i.e., GRAYSCALE image and convert to BGR image
	and if image has 3 channels i.e., BGR image and convert to GRAYSCALE image.

	Return the converted image

	Example:
			image_path = "../images/before/puzzle.jpg"
			pre_process_image = preprocess(image_path)
			converted_image = GRAY_RGB_GRAY(pre_process_image)
			show_image(converted_image)
	"""
    if GRAY_RGB and (len(image.shape) == 2 or image.shape[2] == 1):
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        return image
    else:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image
#-------------------------------------------------------------------------------------------------------
def findContours(image):
    """
    Docstring:
    	Contours:
		    Contours are defined as the line joining all the points along the boundary of an 
		    image that are having the same intensity. Contours come handy in shape analysis, 
		    finding the size of the object of interest, and object detection.
    Return:
    	All the contours in the image

    Example:
			image_path = "../images/before/puzzle.jpg"
			pre_process_image = preprocess(image_path)
			converted_image = GRAY_RGB_GRAY(pre_process_image)
			contours = findContours(converted_image)
    """
    contours,hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours
#-------------------------------------------------------------------------------------------------------
def plotContours(image,showImage = False):
    """
	This Function will plot all contours present in the image, by plotting contours we will get to know
	about the contour present across the suduko grid.

	It will return points of contours.
    """
    contours = findContours(image)
    image = GRAY_RGB_GRAY(image)
    drawContours = cv2.drawContours(image,contours,-1,(226,3,255),3)
    
    if showImage:
        show_image(drawContours)
        
    return drawContours
#-------------------------------------------------------------------------------------------------------
def cal_corners(image):
	"""
	This function will calculate the co-ordinates of suduko grid.
	If we are finding contours of image, and we know that there must be the maximum contours in the 
	image and with probability of 0.99 that msut be of suudko grid and in this function we will 
	calculate the contours and find sorted them in decreasing order and get the biggest contour 
	in the image.

	There are several methods of finding the co-ordinates of suduko, but here I am discussing 2 methods.
	Methods:
		1.approxPolyDP
			perimeter = cv2.arcLength(contour,True)
			epilson = threshold*perimeter
			co_ord = cv2.approxPolyDP(contour,epilson,True)
		2.Ramer Doughlas Peucker algorithm



	Return:
		It will return a list of all 4 co-ordinates of bounding box of suduko grid

			0 -> top_left corner
			1 -> top_right corner
			2 -> bottom_right corner
			3 -> bottom_left corner

	# 1.In approxPolyDP method we approximate the curve given the largest contour. We know that suduko grid 
	#   is RECTANGLE, so there is high probabilty that largest 4 sided contour is our suduko grid.

	#   Function:
	#   	cv2.approxPolyDP(curve,epilson,closed[,approxCurve])

	#   	curve -> The largest Contour
	#   	epilson ->


	"""
    import operator
    
    contours = findContours(image)
    
    contours = sorted(contours,key = cv2.contourArea,reverse=True)
    
    for_largest = contours[0]
    
    bottom_r, _ = max( enumerate([ coordinate[0][0] + coordinate[0][1] for coordinate in for_largest]), key=operator.itemgetter(1) )
    top_l, _ = min( enumerate([ coordinate[0][0] + coordinate[0][1] for coordinate in for_largest]), key=operator.itemgetter(1) )
    bottom_l, _ = min( enumerate([ coordinate[0][0] - coordinate[0][1] for coordinate in for_largest]), key=operator.itemgetter(1) )
    top_r, _ = max( enumerate([ coordinate[0][0] - coordinate[0][1] for coordinate in for_largest]), key=operator.itemgetter(1) )
    
    return [ for_largest[top_l][0],for_largest[top_r][0],for_largest[bottom_r][0],for_largest[bottom_l][0] ]
#-------------------------------------------------------------------------------------------------------
def display_points(image,radius = 10, showImage=False):
	"""
	This Function will display all 4 co-ordinates calculated in cal_corners function
	"""
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
#-------------------------------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------------------------------
def distance(p1,p2):
	"""
	This function will calculate distance between 2 points (p1,p2)
	p1 = (x1,y1)
	p2 = (x2,y2)

	Distance Formula:
		distance = ( ( (x1-x2)**2 ) + ( (y1-y2)**2 ) )**0.5
	"""
    return (np.sum((p1-p2)**2)**0.5)
#-------------------------------------------------------------------------------------------------------
def crop_image(image):
	"""
	Docstring:
		This function will crop the pre_process_image, and get the suduko grid. Firstly, 
		the pre processed image is passed, and by using findContour Function we find the contours,
		then using that contours we will calculate corners of suduko grid, and  around the corners
		we crop our image and finally we get the cropped image i.e., Suduko Grid.
	Args:
		image -> This is a pre_processed image

	Return:
		Image, which is cropped image of pre_processed image and cropped image contains Suduko Grid.

	Example:
		Example:
			image_path = "../images/before/puzzle.jpg"
			pre_process_image = preprocess(image_path)
			cropped_image = crop_image(pre_process_image)
			show_image(cropped_image)
	
	"""
    corners = cal_corners(image)
    
    top_l,top_r,bottom_r,bottom_l = corners
    
    side = max([distance(top_l,top_r),distance(bottom_l,bottom_r),distance(top_l,bottom_l),distance(top_r,bottom_r)])
    
    src = np.array([top_l,top_r,bottom_r,bottom_l],dtype='float32')
    dst = [[0,0], [side - 1,0], [side - 1,side - 1], [0,side - 1]]
    dst = np.array(dst,dtype = 'float32')
    m = cv2.getPerspectiveTransform(src,dst)
    get = cv2.warpPerspective(image,m,(int(side),int(side)))
    return get
#-------------------------------------------------------------------------------------------------------
def grid_box(image,threshold = 5):
	"""
	Docstring:
		This function is calculating co-ordinates of grid of suduko.
		It means that suduko grid has 81 boxes inside i.e., 9 rows and 9 columns. It will calculate the
		the co-ordinates of each grid(square box that contains suduko number or blank image).
		It will return the co-ordinates of each grid.

	Args:
		image -> It takes cropped image as an input.
		threshold -> How much co-ordinates increased by, to avoid noise.

	Return;
		It will return boxes i.e., tuple of points (p1,p2).
		The length of boxes will be 81 because there are 9 rows and 9 columns in suduko puzzle.
		These points (p1,p2) of every grid represent top_left ad bottom_right corner of grid repectively.
	"""
    shape = image.shape[0]
    size = int(shape/9)
    
    boxes = list()
    for i in range(9):
        for j in range(9):
            p1 = (i*size + threshold-3,j*size)
            p2 = ((i+1)*size + threshold-3,(j+1)*size)
            boxes.append((p1,p2))
    return boxes # return top_left and bottom_right co-ordinates
#-------------------------------------------------------------------------------------------------------
def display_grid_box_points(image,radius = 5,color = (226,0,255),showImage = False):
	"""
	Docstring:
		This function plots the points of every grid.
	Args:
		image -> Take input as cropped image.
		radius -> Plotted points are in shape of circle and circle had radius, so it is the radius 
				  of point.
		color -> color of point, it can be tuple of ints or ints.
		showImage -> bool value, if True it will show the points. 

	Return:
		Return an numpy.ndarray of image that contains points.
	"""
    image = crop_image(image)
    points = grid_box(image)
    
    image = GRAY_RGB_GRAY(image)
    
    for pt in points:
        image = cv2.circle(image,pt[0],radius,color, -1)
        image = cv2.circle(image,pt[1],radius,color,-1)
        
    if showImage:
        show_image(image)
    return image
#-------------------------------------------------------------------------------------------------------
def crop_boxes(image,boxes):
	"""
	This Function will crop the grid within the points
	"""
    return image[boxes[0][1]:boxes[1][1], boxes[0][0]:boxes[1][0]]
#-------------------------------------------------------------------------------------------------------
def display_grid_boxes(image,boxes,thickness = 5,showImage = False,colour = (128,255,26)):
    """
	This function makes rectnale boxes around the each suduko grid
	Args:
		image -> Take cropped image as an input.
		boxes -> The co-ordinate of each square grid
		thickness -> the thickness of bounding box i.e., Rectangle around each square grid
		color -> color of rectangle box.

	Return:
		It will return numpy.ndarray of image which contains the rectangle boxes.
    """
    image = GRAY_RGB_GRAY(image)
    for rect in boxes:
        image = cv2.rectangle(image, rect[0],rect[1],colour,thickness)
    if showImage:
        show_image(image)
    return image
#-------------------------------------------------------------------------------------------------------
def generate_color():
    color = list(np.random.choice(range(256),size = 3))
    
    color = tuple(color)
    
    return color
#-------------------------------------------------------------------------------------------------------
def scale_and_centre(img, size, margin=0, background=0):
    h, w = img.shape[:2]
    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))
#-------------------------------------------------------------------------------------------------------
def feature(image,top_l = None,bottom_r = None):
    height,width = image.shape
    
    if top_l is None:
        top_l = [0,0]
    if bottom_r is None:
        bottom_r = [height,width]
        
    total_area = 0.
    seed = (None,None)_
    for w in range(top_l[0],bottom_r[0]):
        for h in range(top_l[1],bottom_r[1]):
            if image[h,w] == 255 and w<width and h<height:
                area = cv2.floodFill(image,None,(w,h),128)
                
                if area[0] > total_area:
                    total_area = area[0]
                    seed = (w,h)
      
    for w in range(width):
        for h in range(height):
            if image.item(h, w) == 255 and w < width and h < height:
                cv2.floodFill(image, None, (w, h), 128)
                
    mask = np.zeros((height + 2, width + 2), np.uint8)
    if all([pt is not None for pt in seed]):
        cv2.floodFill(image, mask, seed, 255)

    top, bottom, left, right = height, 0, width, 0

    for w in range(width):
        for h in range(height):
            if image[h,w] == 128:
                cv2.floodFill(image, mask, (w, h), 0)


            if image[h,w] == 255:
                top = h if h < top else top
                bottom = h if h > bottom else bottom
                left = w if w < left else left
                right = w if w > right else right

    box = [[left, top], [right, bottom]]
    return image, np.array(box, dtype='int64'), seed
#-------------------------------------------------------------------------------------------------------
def extract_digit(image, boxes, size):
	"""
	Docstring:
		This function takes input as cropped image, and return the particular rectangle grid of
		particular size.

	Args:
		image -> Cropped Image
		boxes -> co-ordinates of that particular rectangular grid
		size -> size of image, Recommended to be (28,28) as per MNIST dataset.

	Return:
		It will return the image of that particular grid.
	"""
    digit = crop_boxes(image, boxes) 

    height, width = digit.shape
    margin = int(np.mean([height, width]) / 2.5)
    img, box, seed = feature(digit, [margin, margin], [width - margin, height - margin])
    digit = crop_boxes(digit, box)

    width = box[1][0] - box[0][0]
    height = box[1][1] - box[0][1]


    if width > 0 and height > 0 and (width * height) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
        
    else:
        return np.zeros((size, size), np.uint8)
#-------------------------------------------------------------------------------------------------------
def get_digit(image,boxes,size):
	"""
	Docstring:
		This function will make a list of all 81 rectangular grid, based on cropped image.

	Args:
		image -> Cropped Image
		boxes -> co-ordinates of that particular rectangular grid
		size -> size of image, Recommended to be (28,28) as per MNIST dataset.

	Return:
		It will return the list of all 81 images that are of shape (size,size).
	"""
    images = []
    
    for box in boxes:
        box_image = crop_boxes(image,box)
        images.append(box_image)
        
    digit_image = []
    for box in boxes:
        digit = extract_digit(image,box,size)
        digit_image.append(digit)
        
    return digit_image
#-------------------------------------------------------------------------------------------------------