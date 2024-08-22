import difflib
from termcolor import colored
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import refFont
from enum import Enum
from scipy import stats


class NoiseType(Enum):
    NONE =0             
    GAUSSIAN = 1
    SALT_AND_PEPPER =2
    SPECKLE =3

# TODO: implement the noise fix functions
NoiseFix={
    NoiseType.NONE:None,
    NoiseType.GAUSSIAN:cv2.fastNlMeansDenoising,
    NoiseType.SALT_AND_PEPPER:cv2.medianBlur,
    NoiseType.SPECKLE:cv2.medianBlur
}


TcPans=[
    ["4000","1234","5678","9010"],  # 1
    ["4000","1234","5678","9010"],  # 2
    ["4000","1234","5678","9010"],  # 3
    ["5258","9712","3456","7890"],  # 4
    ["4265","1234","5678","9012"],  # 5
    ["4321","9876","5012","9900"],  # 6
    ["1083","3333","0018","0813"],  # 7
    ["1234","5678","9012","3456"],  # 8
    ["4321","9876","9900","5012"],  # 9
    ["5412","1234","5678","9010"],  # 10
    ["1234","5678","9012","3456"],  # 11
    ["4183","5812","3456","7890"],  # 12
    ["5678","1234","8567","1234"],  # 13
    ["1234","5678","9012","3456"],  # 14
    ["5240","4008","0880","0001"],  # 15
    ["4321","9876","5012","9900"],  # 16
]


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    
    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized


def highlight_differences(str1:list, str2:list)->int:
    """
    Highlights the differences between two strings.
    Characters in str2 that differ from str1 are highlighted in red.
    returns the number of differences and prints the highlighted strings.
    """
    # str1 = sum(str1, [])
    str2 = sum(str2, [])

    str1="".join(str1)
    str2="".join(str2)
    diff = difflib.ndiff(str1, str2)
    highlighted_str1 = []
    highlighted_str2 = []
    differences = 0

    for i, s in enumerate(diff):
        if s[0] == ' ':
            highlighted_str1.append(s[-1])
            highlighted_str2.append(s[-1])
        elif s[0] == '-':
            highlighted_str1.append(s[-1])
            highlighted_str2.append(colored(s[-1], 'red'))
            differences += 1
        elif s[0] == '+':
            highlighted_str1.append(' ')
            highlighted_str2.append(colored(s[-1], 'red'))
            differences += 1

    expected_str        ="expected  : " +"".join(highlighted_str1)
    difference_str      ="difference: " +"".join(highlighted_str2)
    output_str          ="output    : " +"".join(str2)
    text=f"{expected_str}\n{difference_str}\n{output_str}"
    # differences_str     ="expected: " +str(differences)
    return text,differences


def getPan(image,font=refFont.Font(),debug=False):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # image = cv2.imread("E:/vs code projects/CreditCardProject/TC/02.jpg")
    aspectRatio=image.shape[1]/image.shape[0]
    image=cv2.resize(src=image,dsize=(300,int(300//aspectRatio)),interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        showImagesMulti([image,gray],["original","gray"])
    
    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    if debug:
        print("apply tophat morphological operator to find light regions against a dark background")
        showImage(tophat,title="tophat")
    
    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    if debug:
        print("compute gradx of tophat")
        showImage(gradX,title="gradx")
    
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    if debug:
        print("apply closing operation using the rectangular kernel to help close gaps in between credit card number digits")
        showImage(gradX,title="gradx")
        
    thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if debug:
        print("apply Otsu's thresholding method to binarize the gradx")
        showImage(thresh,title="gradx threshed")
        
    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    if debug:
        print("apply a second closing operation to the binary image, again to help close gaps between credit card number regions")
        showImage(thresh,title="gradx threshed")

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    if debug:
        threshCopy=cv2.cvtColor(thresh.copy(),cv2.COLOR_GRAY2BGR)
        print("Number of contours found: ",len(cnts))
        showImage(drawContours(threshCopy,cnts),title="contours")


    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # since credit cards used a fixed size fonts with 4 groups
        # of 4 digits, we can prune potential contours based on the
        # aspect ratio
        if ar > 2.5 and ar < 4.0:
            # contours can further be pruned on minimum/maximum width
            # and height
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                # append the bounding box region of the digits group
                # to our locations list
                locs.append((x, y, w, h))
    # sort the digit locations from left-to-right, then initialize the
    # list of classified digits
    locs = sorted(locs, key=lambda x:x[0])
    output = []
    if debug:
        locsImage=image.copy()
        for x, y, w, h in locs:
            cv2.rectangle(locsImage, (x , y ),(x + w , y + h ), (0, 0, 255), 2)
        showImage(locsImage,title="pan groups")

    # loop over the 4 groupings of 4 digits
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []
        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # detect the contours of each individual digit in the group,
        # then sort the digit contours from left to right
        digitCnts,_ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = sorted(digitCnts, key=lambda x: digitCnts[0][0][0][0])

        """ Debug """
        # filterDigitcContours(digitCnts,group.copy())

        # digitBoundingRects=[cv2.boundingRect(squareContour) for squareContour in digitCnts]
        # digitBoundingRects.extend(digitBoundingRects)
        # groupedContours,_=cv2.groupRectangles(digitBoundingRects,1,0.2)
        # groupCopy=group.copy()
        # groupCopy=cv2.cvtColor(groupCopy,cv2.COLOR_GRAY2BGR)
        # print(f"group {i+1}, number of contours found: {len(digitCnts)}")
        # ut.showImage(ut.drawContours(groupCopy,digitCnts))

        # contourImages=[]

        # for x, y, w, h in groupedContours:
        # 	rectCopy=groupCopy.copy()
        # 	cv2.rectangle(rectCopy, (x , y ),(x + w , y + h ), (0, 0, 255), 2)
        # 	contourImages.append(rectCopy)

        # ut.showImagesMulti(contourImages)
        """ Debug end """
        # loop over the digit contours
        for c in digitCnts:
            # compute the bounding box of the individual digit, extract
            # the digit, and resize it to have the same fixed size as
            # the reference OCR-A images
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            # initialize a list of template matching scores	
            scores = []
            # loop over the reference digit name and digit ROI
            for (digit, digitROI) in font.digits.items():
                # apply correlation-based template matching, take the
                # score, and update the scores list
                # digitROI= cv2.cvtColor(digitROI,cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(roi, digitROI,
                    cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            # the classification for the digit ROI will be the reference
            # digit name with the *largest* template matching score
            groupOutput.append(str(np.argmax(scores)))
        print(groupOutput)


def filterDigitContours(digitContours:list,group,image=None,debug=False)->list:
    """ 
    * digitContours : list of contours of digits
    * group : image of the group of digits.
    """
    
    # showImage(drawContours(image,digitContours))
    imageArea=group.shape[0]*group.shape[1]
    newDigitContours=[]
    for (i, c) in enumerate(digitContours):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = h / float(w)
        # showImage(drawContours(image,[c]))
        flag1=False
        flag2=False
        area=cv2.contourArea(c)
        if ar > 1 and ar < 3:
            flag1=True
            # area=cv2.contourArea(c)
            if area>0.007*imageArea and area<0.3*imageArea:
                flag2=True
                newDigitContours.append(c)

        # if not (flag1 and flag2) and debug:
        #     showImage(drawContours(image,[c]),title="rejected")
        #     print(f"aspect ratio: {ar}, area: {area/imageArea}")
    return newDigitContours

def filterSquares(cnts):
    locs=[]
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # since credit cards used a fixed size fonts with 4 groups
        # of 4 digits, we can prune potential contours based on the
        # aspect ratio
        areaDifference=w*h-cv2.contourArea(c)
        if areaDifference>0.2*w*h:
            continue

        if ar > 0.9 and ar < 4.0:
            # contours can further be pruned on minimum/maximum width
            # and height
            if (w > 40) and (h > 10):
                # append the bounding box region of the digits group
                # to our locations list
                locs.append(c)
    return locs


def drawRect(image,rect):
    """ 
    draws a rectangle on an image
    """
    x, y, w, h = rect
    return cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)


def displayRectsIndividually(image,rects):
    """ 
    displays the rectangles individually
    """
    for rect in rects:
        imageCopy=drawRect(image,rect)
        showImage(imageCopy)
    return imageCopy


def displayContoursIndividually(image,cnts):
    """ 
    displays the contours individually
    """
    for cnt in cnts:
        imageCopy=drawContours(image,[cnt])
        showImage(imageCopy)
    return imageCopy


def displayRects(image,rects):
    """ 
    displays the rectangles
    """
    for rect in rects:
        image=drawRect(image,rect)
    showImage(image)
    return image

def displayRect(image,rect):
    """ 
    displays the rectangle
    """
    rectImage=drawRect(image,rect)
    showImage(rectImage)
    return rectImage


def filterPanGroups(cnts,image=None):
    locs=[]
    """ 
    debug
    """
    #* rects=[cv2.boundingRect(c) for c in cnts]
    #* displayRects(image,rects)
    """ 
    debug
    """
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # since credit cards used a fixed size fonts with 4 groups
        # of 4 digits, we can prune potential contours based on the
        # aspect ratio
        """ 
        debug
        """
        # * displayRect((x,y,w,h),image)
        # *flag1=flag2=False
        """ 
        debug
        """
        if ar > 2.5 and ar < 4.0:
            # contours can further be pruned on minimum/maximum width
            # and height
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                # append the bounding box region of the digits group
                # to our locations list
                locs.append((x, y, w, h))

        #* if flag1 is False or flag2 is False:
            #* displayRect(image,(x,y,w,h))
    return locs


def getPanUpdated(image,font=refFont.Font(),debug=False,digitGroupDebug=False):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # image = cv2.imread("E:/vs code projects/CreditCardProject/TC/02.jpg")
    aspectRatio=image.shape[1]/image.shape[0]
    image=cv2.resize(src=image,dsize=(300,int(300//aspectRatio)),interpolation=cv2.INTER_LINEAR)
    if len(image.shape)==2:
        gray=image

    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if debug:
        showImagesMulti([image,gray],["original","gray"])
    
    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    if debug:
        print("apply tophat morphological operator to find light regions against a dark background")
        showImage(tophat,title="tophat")
    
    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    if debug:
        print("compute gradx of tophat")
        showImage(gradX,title="gradx")
    
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    if debug:
        print("apply closing operation using the rectangular kernel to help close gaps in between credit card number digits")
        showImage(gradX,title="gradx")
        
    thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if debug:
        print("apply Otsu's thresholding method to binarize the gradx")
        showImage(thresh,title="gradx threshed")
        
    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    threshCopy = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    if debug:
        print("apply a second closing operation to the binary image, again to help close gaps between credit card number regions")
        showImage(threshCopy,title="gradx threshed")

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs = []
    if debug or digitGroupDebug:
        threshCopy=cv2.cvtColor(thresh.copy(),cv2.COLOR_GRAY2BGR)
        print("Number of contours found: ",len(cnts))
        showImage(drawContours(threshCopy,cnts),title="contours")

    locs = filterPanGroups(cnts,image)
    if len(locs)<4:     # special case when
        threshTest = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rectKernel)
        cnts,_ = cv2.findContours(threshTest.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        locs = filterPanGroups(cnts,image)

    # sort the digit locations from left-to-right, then initialize the
    # list of classified digits
    locs = sorted(locs, key=lambda x:x[0])
    output = []
    if debug or digitGroupDebug:
        locsImage=image.copy()
        for x, y, w, h in locs:
            cv2.rectangle(locsImage, (x , y ),(x + w , y + h ), (0, 0, 255), 2)
        showImage(locsImage,title="pan groups")

    # loop over the 4 groupings of 4 digits
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        
        # detect the contours of each individual digit in the group,
        # then sort the digit contours from left to right

        """ 
        debug
        """
        for i in range(4):              # trails
            groupOutput = []
            group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5].copy()
            group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            groupCopy=group.copy()
            flag=False
            if i==1:
                ksize = 3
                # Define the kernel
                kernel = np.ones((ksize, ksize), np.uint8)
                # printprint(f"before morph")
                # showImage(group)
                group=resize_with_aspect_ratio(group,width=group.shape[1]*2)
                groupCopy = cv2.morphologyEx(group      , cv2.MORPH_CLOSE, kernel)
                # groupCopy = cv2.morphologyEx(groupCopy  , cv2.MORPH_CLOSE, kernel)
                # groupCopy = cv2.morphologyEx(groupCopy  , cv2.MORPH_CLOSE, kernel)
                # printprint(f"after morph")
                # showImage(groupCopy)

            if i==2:
                ksize = 3
                # Define the kernel
                kernel = np.ones((ksize, ksize), np.uint8)
                # printprint(f"before morph")
                # showImage(group)
                group=resize_with_aspect_ratio(group,width=group.shape[1]*2)
                groupCopy = cv2.morphologyEx(group      , cv2.MORPH_CLOSE, kernel)
                groupCopy = cv2.morphologyEx(groupCopy  , cv2.MORPH_CLOSE, kernel)
                # groupCopy = cv2.morphologyEx(groupCopy  , cv2.MORPH_CLOSE, kernel)
                # printprint(f"after morph")
                # showImage(groupCopy)

            if i==3:
                ksize = 3
                # Define the kernel
                kernel = np.ones((ksize, ksize), np.uint8)
                # printprint(f"before morph")
                # showImage(group)
                group=resize_with_aspect_ratio(group,width=group.shape[1]*2)
                groupCopy = cv2.morphologyEx(group      , cv2.MORPH_CLOSE, kernel)
                groupCopy = cv2.morphologyEx(groupCopy  , cv2.MORPH_CLOSE, kernel)
                groupCopy = cv2.morphologyEx(groupCopy  , cv2.MORPH_CLOSE, kernel)
                # printprint(f"after morph")
                # showImage(groupCopy)
            """ 
            debug
            """

            digitCnts,_ = cv2.findContours(groupCopy.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            digitCnts = sorted(digitCnts, key=lambda x: digitCnts[0][0][0][0])
            if debug or digitGroupDebug:
                print(f"number of contours: {len(digitCnts)}")
                showImage(drawContours(image[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5],digitCnts))

            digitCntsRects=digitCnts.copy()

            #* print(f"group {i+1} contours")
            #* displayContoursIndividually(image[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5],digitCnts)
            digitCnts=filterDigitContours(digitCnts,groupCopy.copy(),image[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5],debug=debug)

            digitCnts = sorted(digitCnts, key=lambda x: cv2.boundingRect(x)[0])
            
            # * print(f"group {i+1} contours")
            # * displayContoursIndividually(image[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5],digitCnts)

            digitBoundingRects=[cv2.boundingRect(squareContour) for squareContour in digitCntsRects]

            #* print(f"group {i+1} rects")
            #* displayRectsIndividually(image[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5],digitBoundingRects)
            digitBoundingRects.extend(digitBoundingRects)
            groupedRects,_=cv2.groupRectangles(digitBoundingRects,1,0.6)
            # cv2.groupRectangles()
            imageCopy=cv2.cvtColor(groupCopy.copy(),cv2.COLOR_GRAY2BGR)
            if debug or digitGroupDebug:
                
                print(f"group {i+1}, number of digits found using contours: {len(digitCnts)}")
                showImage(drawContours(imageCopy,digitCnts))
            if debug or digitGroupDebug:
                
                print(f"number of digits found using rectangles grouping: {len(groupedRects)}")
                contourImages=[]
                for x, y, w, h in groupedRects:
                    rectCopy=imageCopy.copy()
                    cv2.rectangle(rectCopy, (x , y ),(x + w , y + h ), (0, 0, 255), 2)
                    contourImages.append(rectCopy)
                if len(contourImages)>1:
                    showImagesMulti(contourImages)
                elif len(contourImages)==1:
                    showImage(contourImages[0])

            # loop over the digit contours
            for c in digitCnts:
                # compute the bounding box of the individual digit, extract
                # the digit, and resize it to have the same fixed size as
                # the reference OCR-A images
                (x, y, w, h) = cv2.boundingRect(c)
                roi = groupCopy[y:y + h, x:x + w]
                #* print("templateMatching")
                #* showImage(roi)
                roi = cv2.resize(roi, (57, 88))
                # initialize a list of template matching scores	
                scores = []
                # loop over the reference digit name and digit ROI
                for (digit, digitROI) in font.digits.items():
                    # apply correlation-based template matching, take the
                    # score, and update the scores list
                    # digitROI= cv2.cvtColor(digitROI,cv2.COLOR_BGR2GRAY)
                    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)
                # the classification for the digit ROI will be the reference
                # digit name with the *largest* template matching score
                groupOutput.append(str(np.argmax(scores)))
            # print(groupOutput)
            if len(groupOutput)==4:
                output.append(groupOutput)
                break
    return output


def _showPlt(image):
    if len(image.shape) == 3:    # means that the image is colored
        plt.imshow(image)
    elif len(image.shape) == 2:         # means that the image is grayscale
        plt.imshow(image,cmap='gray')    


def showImagesMulti(images,titles=None):
    _, axarr = plt.subplots(nrows=1, ncols=len(images), figsize=(20,10)) # figsize is in inches, yuck
    for i,image in enumerate(images):
        plt.sca(axarr[i]); plt.title(titles[i] if titles is not None else "image"); _showPlt(image);

    plt.show()


def printWindow(image, point , window_size=5):
    # Get the dimensions of the image
    x,y=point
    
    height, width = image.shape[:2]

    print(f"point: ({x},{y}) , value : {image[y, x]}")
    # Calculate the starting and ending indices for the window
    start_x = max(0, x - window_size // 2)
    start_y = max(0, y - window_size // 2)
    end_x = min(width - 1, x + window_size // 2)
    end_y = min(height - 1, y + window_size // 2)

    # Extract the window from the image
    for j in range(start_y, end_y + 1):
        for i in range(start_x, end_x + 1):
            if i == x and j == y:
                print(f"({i},{j}): {image[j, i]} (center)")
            else:
                print(f"({i},{j}): {image[j, i]}")
        print()  # Move to the next line for the next row of pixels


def drawCornerPoints(image,corners):
    # Filter out perfect 90-degree corners
    filtered_corners = []
    for corner in corners:
        x, y = corner.ravel()
        filtered_corners.append((x, y))

    # Visualize the detected corners on the original image
    for corner in filtered_corners:
        x, y = corner
        cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

    return image


def getImagePathsInDirectory(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_paths.append(os.path.join(root, file))
    return image_paths


def _showPlt(image):
    """ 
    private function
    """
    if len(image.shape) == 3:    # means that the image is colored
        plt.imshow(image)
    elif len(image.shape) == 2:         # means that the image is grayscale
        plt.imshow(image,cmap='gray')    


def showImagesMulti(images,titles=None):
    """ 
    displays multiple images on the same row
    """
    _, axarr = plt.subplots(nrows=1, ncols=len(images), figsize=(20,10)) # figsize is in inches, yuck
    for i,image in enumerate(images):
        plt.sca(axarr[i]); plt.title(titles[i] if titles is not None else "image"); _showPlt(image);

    plt.show()

def isAspectRatioOfNumber(aspectRatio)->bool:
    """ 
    checks if the aspect ratio is close to 1.5 which is the aspect ratio of a number
    """
    return aspectRatio<1.6 and aspectRatio>1.4 

def showImage(image, title:str="image",cmap=None)->None:
    plt.figure(figsize=(3, 3))
    if cmap is not None:
        plt.imshow(image,cmap=cmap)
    elif len(image.shape) == 3:    # means that the image is colored
        plt.imshow(image)
    elif len(image.shape) == 2:         # means that the image is grayscale
        plt.imshow(image,cmap='gray')    
    plt.title(title)
    plt.show()


def drawContours(image, contours):
    """ 
    draw the outline of a contour contours on the bgr image
    """
    newImage=image.copy()
    for i in range(len(contours)):
        cv2.drawContours(newImage,contours,i,(0,255,0),4)
        i = i + 1
    return newImage


def drawContourPoints(image, contours,isPoints=False):
    """ 
    draws the points of the vertices of a contour
    """
    # Iterate over each contour
    if type(contours) is not list and type(contours) is not tuple:
        if not isPoints:
            for contour in contours:
                for point in contour:
                    # Extract x and y coordinates of the point
                    x, y = point[0]
                    # Draw a point (circle) on the image at the current point
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Red color, filled circle
        else:
            for point in contours:
                # Extract x and y coordinates of the point
                x, y = point
                # Draw a point (circle) on the image at the current point
                cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red color, filled circle
    else:
        if isPoints:
            for i,point in enumerate(contours):
                # Extract x and y coordinates of the point
                x, y = point[0]
                
                # Draw a point (circle) on the image at the current point
                cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Red color, filled circle
        else:
            for j,contour in enumerate(contours):
                # Iterate over each point in the contour
                for i,point in enumerate(contour):
                    # Extract x and y coordinates of the point
                    x, y = point[0]
                    
                    # Draw a point (circle) on the image at the current point
                    cv2.circle(image, (x, y), 10, (0, 0, 255), -1)  # Red color, filled circle

    return image

def drawContourPixels(image, contours):
    """ 
    colors the vertices pixels of a contour
    """
    # Iterate over each contour
    if type(contours) is not list and type(contours) is not tuple:
        for point in contours:
            # Extract x and y coordinates of the point
            x, y = point[0]
            # Draw a point (circle) on the image at the current point
            image[y, x] = (0, 0, 255)  # Red color, filled circle
    else:
        for contour in contours:
            # Iterate over each point in the contour
            for point in contour:
                # Extract x and y coordinates of the point
                x, y = point[0]
                # Draw a point (circle) on the image at the current point
                image[y, x] = (0, 0, 255)  # Red color, filled circle

    return image


def orderPoints(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]  # topleft
    new_points[3] = points[np.argmax(add)] # bottomright

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]     # topright
    new_points[2] = points[np.argmax(diff)]    # bottomleft

    return new_points


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


def computeCenter(contours):
    """ 
    computes the center for the combined contour
    """
    # Combine contours into one array
    if len(contours)==0:
        return None
    
    combined_contour = np.vstack(contours)

    # Compute bounding box for the combined contours
    x, y, w, h = cv2.boundingRect(combined_contour)

    # Calculate center point of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    return (center_x, center_y)


def find_missing_point(point1, point2, known_point):
    """ 
    finds the missing point in a line that is parallel to another line and the same lenght
    """
    point1 = list(point1.ravel())
    point2 = list(point2.ravel())
    known_point = list(known_point.ravel())
    # Determine the direction vector of the first line
    direction_vector1 = np.array(point2) - np.array(point1)
    
    # Find the equation of the first line (y = mx + c)
    m1 = direction_vector1[1] / direction_vector1[0]  # Slope
    c1 = point1[1] - m1 * point1[0]  # Intercept
    
    
    # Find the equation of the second line
    m2 = m1  # Lines are parallel, so slopes are equal
    c2 = known_point[1] - m2 * known_point[0]  # Intercept

    # Calculate the length of the vector
    vector_length = np.linalg.norm(direction_vector1)
    
    # Find the x-coordinate of the intersection point
    intersection_x = known_point[0] + vector_length*np.cos(np.arctan(m1))
    
    # Find the y-coordinate of the intersection point
    intersection_y = known_point[1] + vector_length*np.sin(np.arctan(m1))
    
    return intersection_x, intersection_y


def biggestContours(contours,numberOfContours=1):
    # Sort the contours by area in descending order
    contours = [np.array(contour) for contour in contours]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:numberOfContours]
    return contours


def resize_image(image, target_width, target_height):
    # Calculate the aspect ratio of the image
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    # Calculate the new dimensions while maintaining aspect ratio
    if target_width is None:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    elif target_height is None:
        new_height = int(target_width / aspect_ratio)
        new_width = target_width
    else:
        new_width = target_width
        new_height = target_height

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def fixSkew(img,angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def fixSinsoidalNoise(gray, debug=False):
    # Compute fourier tranform of the gray
    f_transform = np.fft.fft2(gray)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # # View transform and zoom on the middle
    magnitude_spectrum =  np.log(np.abs(f_transform_shifted) +1)
    # magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    # plt.imshow(magnitude_spectrum, cmap='gray')

    # Create a filter to remove sinusoidal noise of frequency 11
    mask = np.ones_like(gray)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask[crow][ccol-5] = 0
    mask[crow][ccol+5] = 0
    # mask[crow][ccol-6] = 0
    # mask[crow][ccol+6] = 0


    # Apply the filter mask
    f_transform_shifted_filtered = f_transform_shifted * mask

    # # View transform and zoom on the middle
    magnitude_spectrum_filtered =  np.log(np.abs(f_transform_shifted_filtered) + 1)
    # magnitude_spectrum_filtered = cv2.normalize(magnitude_spectrum_filtered, None, 0, 255, cv2.NORM_MINMAX)


    # Do inverse fourier transform
    fft_ifft_shift = np.fft.ifftshift(f_transform_shifted_filtered)
    imageThen = np.fft.ifft2(fft_ifft_shift)
    imageThen = np.abs(imageThen)
    # plt.imshow(imageThen, cmap='gray')

    # Apply thresholding
    ret, thresholded_spectrum = cv2.threshold(imageThen, 190, 255, cv2.THRESH_BINARY)
    return (imageThen,thresholded_spectrum)


def detect_noise_type(image_gray, threshold=0.05)->NoiseType:
    # Calculate the standard deviation of pixel intensities
    std_dev = np.std(image_gray)

    # Check for Gaussian noise based on standard deviation
    if std_dev < threshold * 255:
        return NoiseType.GAUSSIAN

    # Calculate the histogram of pixel intensities
    histogram = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

    # Check for salt and pepper noise based on histogram spikes
    peak_ratio = np.sum(histogram[:10]) + np.sum(histogram[-10:])
    total_pixels = image_gray.shape[0] * image_gray.shape[1]
    if peak_ratio / total_pixels > 0.01:
        return NoiseType.SALT_AND_PEPPER

    # Check for speckle noise based on local variance
    variance = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    if variance < threshold:
        return NoiseType.SPECKLE

    return NoiseType.NONE


def isSkewed(gray)->float:
    cnts,_ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts=filterSquares(cnts)

    minTheta = 0

    angles=[]
    for cnt in cnts:
        (x,y),(width,height),theta = cv2.minAreaRect(cnt)
        angles.append(theta)

    angle,_ = stats.mode(angles)
    angle-=90
    if angle<1:              # if the mode is less than 1 degree then the image is not skewed
        return 0
    
    return angle


def fixPerspective(image,debug= False):
    grey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grey, (7, 7), 0)
    canny=cv2.Canny(blurred.copy(),50,150)
    externalCannyContours,heirarchy = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        cannyImage=cv2.cvtColor(canny.copy(),cv2.COLOR_GRAY2BGR)

    newexternalCannyContours=[]
    for contour in externalCannyContours:
        # Approximate contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has 4 vertices
        if len(approx) == 4:
            newexternalCannyContours.append(approx)
            if debug:
                cv2.drawContours(cannyImage, [approx], -1, (0, 255, 0), 2)
                showImage(cannyImage)

    if len(newexternalCannyContours)==0:
        return image
    
    width,height=(1146,721)
    # h,w=image.shape[:2]
    h,w=(height,width)
    topLeft,topRight,bottomLeft,bottomRight=orderPoints(max(newexternalCannyContours, key=cv2.contourArea))  # get corner points for the largest contour
    pts1 = np.float32([topLeft.ravel(),topRight.ravel() ,bottomLeft.ravel() ,bottomRight.ravel()])
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    output = cv2.warpPerspective(image, matrix, (width,height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return output


def removeBackground(thresh,offset=0):
    start_row = -1
    start_col = -1
    end_row = -1
    end_col = -1
    
    width, height = thresh.shape

    background=sorted(thresh[ 0:int(0.1*width) , height//2 ])   # get the background color from middle left of the image
    background= background[len(background)//2]  # get the middle value of the background color
    
    foreground=255-background
    
    for row_index, row in enumerate(thresh):
        for pixel in row:
            if pixel != background:
                start_row = row_index
                break
        if start_row != -1:
            break

    for row_index, row in enumerate(thresh[::-1]):
        for pixel in row:
            if pixel != background:
                end_row = thresh.shape[0] - row_index
                break
        if end_row != -1:
            break

    for col_index, col in enumerate(cv2.transpose(thresh)):
        for pixel in col:
            if pixel != background:
                start_col = col_index
                break
        if start_col != -1:
            break

    for col_index, col in enumerate(cv2.transpose(thresh)[::-1]):
        for pixel in col:
            if pixel != background:
                end_col = thresh.shape[1] - col_index
                break
        if end_col != -1:
            break
    qr_no_quiet_zone = thresh[start_row-offset:end_row+offset, start_col-offset:end_col+offset]
    return qr_no_quiet_zone, (start_row-offset, end_row+offset, start_col-offset, end_col+offset)
