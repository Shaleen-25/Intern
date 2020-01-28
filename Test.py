
'''
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

im = Image.open('img2.png')
data= pytesseract.image_to_data(Image.open('img2.png'), output_type='data.frame')

print (data['0']['height'])


#print(data.columns)
#print(data.loc[0])

'''

#img3 has 90 for Amex Ethics Hotline
import pytesseract
from pytesseract import Output
import cv2
from PIL import Image
import pandas as pd




d = pytesseract.image_to_data(Image.open('full.png'),output_type=Output.DICT)

left = [];
top =[];
text =[];

for i in range(len(d['level'])):
    if(int(d['conf'][i]) >83):
        left.append(d['left'][i])
        top.append((d['top'][i]))
        text.append((d['text'][i]))


Text_Data= pd.DataFrame(
    {'Left': left,
     'Top': top,
     'Text': text
    })

#Text_Data.to_csv("file.txt" , encoding='utf-8')

Text_Data.to_json("full.json",orient='columns')

print("done")

# with open('file.txt', mode = 'w') as f:
#     f.write(left)
#     f.write(top)
#     f.write(text)

# d = pytesseract.image_to_data(img)
print


