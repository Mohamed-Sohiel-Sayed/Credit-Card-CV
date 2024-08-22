import cv2 
import generate_Ref_Digits as ref
import utils as ut
import numpy as np

class Font:
    def __init__(self):
        imagesPaths=ref.run()
        self.digits={}
        
        for i,imagePath in enumerate(imagesPaths):
            image=cv2.imread(imagePath)
            grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            _,grey = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY)
            contour,_ = cv2.findContours(grey, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)       # get contour of the number
            
            # compute the bounding box for the digit, extract it, and resize
            # it to a fixed size
            (x, y, w, h) = cv2.boundingRect(contour[0])
            roi = grey[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            # update the digits dictionary, mapping the digit name to the ROI
            self.digits[i] = roi


    def getDigits(self)->dict:
        return self.digits

if __name__=="__main__":
    font=Font()