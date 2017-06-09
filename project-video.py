#!/usr/bin/env python
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML

(y1-y2)/(x1-x2)=b
(y1-y2)=b(x1-x2)
y1=(b(x1-x2))+y2
y2=-1((b(x1-x2))-y1)
1(500-1)+1

(x1-x2) = (b/(y1-y2))
x1 = (b/(y1-y2)) + x2
x2 = -1((b/(y1-y2))-x1)
y1 = (b(x1-x2))+y2
y2 = -1((b(x1-x2))-y1)

slope = lambda x1,y1,x2,y2: (y1-y2)/(x1-x2)
newx1 = lambda b,y1,x2,y2: ((y1-y2)/b)+x2
newx2 = lambda b,x1,y1,y2: -1*(((y1-y2)/b)-x1)
newy1 = lambda b,x1,x2,y2: (b*(x1-x2))+y2
newy2 = lambda b,x1,y1,x2: -1*((b*(x1-x2))-y1)

(376-600)/(593-)


getx = lambda b,y1,x2,y2: ((y1-y2)/b)+x2
