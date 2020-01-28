import cv2 as cv
import PIL
from PIL import Image 
import numpy as np
import skimage
from skimage import data
import matplotlib.pyplot as plt

img = Image.open("Desktop/button.png").convert("RGB")

plt.figure(figsize=(30,30))
plt.imshow(cpy)

cpy=np.array(img)

np.shape(cpy)
mask=np.array([0,111,207])

cpy[140,23]

for i in range (2652):
    for j in range (1406):
        kt= np.array(cpy[j,i])
        if(kt.all()!=mask.all()):
            cpy[j,i]=[255,255,255]
            


plt.figure(figsize=(30,30))
plt.imshow(cpy)


def border(x, y, w, h):
    draw.rectangle(((y,x), (h,w)) , outline = "black")

import pytesseract
from skimage import color
from pytesseract import Output
from PIL import Image , ImageDraw, ImageFont
import cv2
img= Image.open("desktop/base_img.png")

#imgg = color.xyz2rgb(img)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("Desktop/Roboto-Light.ttf", 30)
d2 = pytesseract.image_to_data(img, output_type=Output.DICT)
im = img.load()
#print (im[23,234])

n_boxes = len(d2['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d2['left'][i], d2['top'][i], d2['width'][i], d2['height'][i])
    if(int(d2['conf'][i])>85):
#         a = d2['left'][i]-5
#         b = d2['top'][i]-5
#         clr = im[a,b]
          border(y,x,y+h,x+w)
          draw.text((d2['left'][i], d2['top'][i]-20), text = d2['text'][i], font = font,  fill = "black")
        #print(d2['text'][i])
        

plt.figure(figsize=(100,100))
plt.imshow(img)




def border(x, y, w, h, clr):
    draw.rectangle(((y,x), (h,w)) , fill =(clr))

import pytesseract
from skimage import color
from pytesseract import Output
from PIL import Image , ImageDraw, ImageFont
import cv2
img= Image.open("desktop/full.png")

#imgg = color.xyz2rgb(img)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("Desktop/Roboto-Light.ttf", 30)
d2 = pytesseract.image_to_data(img, output_type=Output.DICT)
im = img.load()
#print (im[23,234])

n_boxes = len(d2['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d2['left'][i], d2['top'][i], d2['width'][i], d2['height'][i])
    if(int(d2['conf'][i])>83):
        a = d2['left'][i]-5
        b = d2['top'][i]-5
        clr = im[a,b]
        border(y,x,y+h,x+w,clr)
#       draw.text((d2['left'][i], d2['top'][i]-20), text = d2['text'][i], font = font,  fill = "black")
        #print(d2['text'][i])
        

img.save("lele2.png")

plt.figure(figsize=(100,100))
plt.imshow(img)


import skimage
from skimage import data
from PIL import Image , ImageDraw
import matplotlib.pyplot as plt
import numpy as np


img = np.random.random([300,300])
img

img.shape

plt.imshow(img,cmap='gray',interpolation ='nearest')

cam=data.camera()
print (type(cam))
print (cam.shape)
print (cam.size)
plt.imshow(cam)

cam.min(), cam.max(), cam.mean()
img = Image.open("Desktop/button.png").convert("RGB")


tree= skimage.io.imread("Desktop/button.png")
tree.shape

plt.figure(figsize=(9,8))
plt.imshow(tree)

#since img is loaded as a numpy array, we can tweak pixel values as elements of an array
tree_copy=tree.copy()
tree_copy[127,1111]

rgb color scheme img, converting green pixels with intensity>100 to white
mask= tree_copy[:,:,1]>100

ntree=tree.copy()
ntree[mask]=[255,255,255]
plt.figure(figsize=(9,8))
plt.imshow(ntree)


prt_zoom=tree_copy[55:150,350:450]
plt.imshow(prt_zoom)

#rgb to gray
from skimage import color
gtree= color.rgb2gray(tree)
plt.figure(figsize=(9,8))
plt.imshow(gtree,cmap="gray")

#saving an image
skimage.io.imsave("gtree.jpg", gtree)

#drawing line
from skimage import io,draw
x,y = draw.line(200,0,350,350)
tree[x, y] = 1
io.imshow(tree)

drawing rectange
import numpy as np
from PIL import Image , ImageDraw
tree= skimage.io.imread("desktop/Parrot.jpg")
img = Image.open("desktop/Parrot.jpg")
draw = ImageDraw.Draw(img)

def border(x, y, w, h):
    #tree[draw.line(x,y,w,y)]=1
    #tree[draw.line(w,y,w,h)]=1
    #tree[draw.line(x,h,w,h)]=1
    #tree[draw.line(x,y,x,h)]=1
    draw.rectangle(((y,x), (h+1,w)) , outline ="black")
    
border(55,380,150,420)
plt.figure(figsize=(11,11))
plt.imshow(img)


#rotating,shifting,scaling
from skimage.transform import rotate
img_rot = rotate(tree, 10)
io.imshow(img_rot)


#drawing rectange
import numpy as np
from PIL import Image , ImageDraw, ImageFont
tree= skimage.io.imread("desktop/img2.png")
img = Image.open("Desktop/img2.png")

draw = ImageDraw.Draw(img)

def border(x, y, w, h):
    #tree[draw.line(x,y,w,y)]=1
    #tree[draw.line(w,y,w,h)]=1
    #tree[draw.line(x,h,w,h)]=1
    #tree[draw.line(x,y,x,h)]=1
    draw.rectangle(((y,x), (h,w)) , outline ="black")
    

border(27,23,(27+21),(23+134))
#y,x;y+h,x+w
#left	top	width	height
#x  y .  w . h
#23	27	134	21
border(27,167,(27+21),(167+152))
#167	27	152	21
plt.figure(figsize=(22,22))
plt.imshow(img)


def border(x, y, w, h):
    draw.rectangle(((y,x), (h,w)) , outline ="black")

import pytesseract
from skimage import color
from pytesseract import Output
from PIL import Image , ImageDraw, ImageFont
import cv2
img= Image.open("desktop/base_img.png")
imgF= Image.open("desktop/edit_img.png")
#imgg = color.xyz2rgb(img)
draw = ImageDraw.Draw(imgF)
font = ImageFont.truetype("Desktop/Roboto-Light.ttf", 20)
d = pytesseract.image_to_data(img, output_type=Output.DICT)
d2= pytesseract.image_to_data(imgF, output_type=Output.DICT)
n_boxes = len(d2['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d2['left'][i], d2['top'][i], d2['width'][i], d2['height'][i])
    if(d['text'][i]!=d2['text'][i]):
        border(y,x,y+h,x+w)
        draw.text((d2['left'][i], d2['top'][i]-20), text = d['text'][i], font = font,  fill = "black")

plt.figure(figsize=(120,120))
plt.imshow(imgF)


import pytesseract
from skimage import color
from pytesseract import Output
from PIL import Image, ImageFilter,ImageOps
import cv2
img= (Image.open("desktop/img5.png").convert("L"))
img = ImageOps.invert(img)
print(np.shape(img))


font = ImageFont.truetype("Desktop/Roboto-Light.ttf", 20)
 

# Find the edges by applying the filter ImageFilter.FIND_EDGES

#img = img.filter(ImageFilter.FIND_EDGES)
draw = ImageDraw.Draw(img)
 

# display the original show

d = pytesseract.image_to_data(img, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #print(d['text'][i])
    if(int(d['conf'][i]) > 80):
        border(y,x,y+h,x+w)
        draw.text((d['left'][i], d['top'][i]-20), text = d['text'][i], font = font,  fill = "white")

plt.figure(figsize=(50,50))
plt.imshow(img,cmap="gray")


import cv2

method = cv2.TM_SQDIFF_NORMED

# Read the images from the file
small_image = cv2.imread('Desktop/small.png')
large_image = cv2.imread('Desktop/large_image.jpeg')

result = cv2.matchTemplate(small_image, large_image, method)

# We want the minimum squared difference
mn,_,mnLoc,_ = cv2.minMaxLoc(result)

# Draw the rectangle:
# Extract the coordinates of our best match
MPx,MPy = mnLoc

# Step 2: Get the size of the template. This is the same size as the match.
trows,tcols = small_image.shape[:2]

# Step 3: Draw the rectangle on large_image
cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

# Display the original image with the rectangle around the match.
cv2.imshow('output',large_image)

# The image is only displayed if we call this
cv2.waitKey(0)

def border(x, y, w, h):
    draw.rectangle(((y,x), (h,w)) , outline ="black")

import pytesseract
from skimage import color
from pytesseract import Output
from PIL import Image , ImageDraw, ImageFont
import cv2
img= Image.open("desktop/ok.png")

#imgg = color.xyz2rgb(img)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("Desktop/Roboto-Light.ttf", 20)
d2 = pytesseract.image_to_data(img, output_type=Output.DICT)

n_boxes = len(d2['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d2['left'][i], d2['top'][i], d2['width'][i], d2['height'][i])
    if(int(d2['conf'][i])>85):
        border(y,x,y+h,x+w)
        draw.text((d2['left'][i], d2['top'][i]-20), text = d2['text'][i], font = font,  fill = "black")
        
plt.figure(figsize=(120,120))
plt.imshow(img)



#level	page_num	block_num	par_num	line_num	word_num




def border(x, y, w, h):
    draw.rectangle(((y,x), (h,w)) , outline ="black")

import pytesseract
from skimage import color
from pytesseract import Output
from PIL import Image , ImageDraw, ImageFont
import cv2
img= Image.open("desktop/ok.png")

#imgg = color.xyz2rgb(img)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("Desktop/Roboto-Light.ttf", 20)
d2 = pytesseract.image_to_data(img, output_type=Output.DICT)

n_boxes = len(d2['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d2['left'][i], d2['top'][i], d2['width'][i], d2['height'][i])
    if(int(d2['conf'][i])>85):
        border(y,x,y+h,x+w)
        draw.text((d2['left'][i], d2['top'][i]-20), text = d2['text'][i], font = font,  fill = "black")
        
plt.figure(figsize=(120,120))
plt.imshow(img)





            