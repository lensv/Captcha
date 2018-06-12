import random
# import matplotlib.pyplot as plt
import string
import numpy as np
import sys
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt


filename="F:/project/tensorflow验证码/captcha/"
#字体的位置，不同版本的系统会有不同BuxtonSketch.ttf
#font_path = r'C:/Windows/Fonts/RobotoCondensed-Regular.ttf'
font_path = r'C:/Windows/Fonts/l_10646.ttf'

#生成几位数的验证码
num = 4
#生成验证码图片的高度和宽度
size = (160,60)
#背景颜色，默认为白色
#bgcolor = random.choice(((175,111,195),(124,179,182),(105,111,129),(241,195,243),(244,220,229),(219,229,226),(224,215,233)))

#字体颜色，默认为蓝色
#fontcolor = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

#是否要加入干扰线
draw_line = True
#加入干扰线条数的上下限
line_number = (1,5)

#用来随机生成一个字符串
# def gene_text():
#     # source = list(string.letters)
#     # for index in range(0,10):
#     #     source.append(str(index))
#     source = ['0','1','2','3','4','5','6','7','8','9']
#     source = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I','J', 'K','L', 'M', 'N','O','P','Q','R',
#                'S', 'T', 'U', 'V', 'W', 'Z','X', 'Y']
#     return ''.join(random.sample(source,number))#number是生成验证码的位数
#


# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']



#生成四个字符， 验证码一般都无视大小写；验证码长度4个字符
#返回一个含有四个字符串的列表
def random_captcha_text(char_set=number #+ alphabet
                                 + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)#choice()方法返回一个列表，元组或字符串的随机项
        captcha_text.append(c)
    return captcha_text



# captcha_text = random_captcha_text()#生成4个字符串
# captcha_text = ''.join(captcha_text)#将4个字符串合在一起变成一个字符串




#用来绘制干扰线
def gene_line(draw,width,height):
    line_num = random.randint(*line_number)  # 干扰线条数

    for i in range(line_num):
        # 起始点
        begin = (random.randint(0, size[0]), random.randint(0, size[1]))
        # 结束点
        end = (random.randint(0, size[0]), random.randint(0, size[1]))
        #begin = (0, random.randint(0, height))
        #end = (74, random.randint(0, height))
        # 干扰线颜色。默认为红色
        linecolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.line([begin, end], fill = linecolor,width=4)




#生成验证码
def gen_captcha_text_and_image():
    width,height = size #宽和高
    # 背景颜色，默认为白色
    bgcolor = random.choice(((175, 111, 195), (124, 179, 182), (105, 111, 129), (241, 195, 243), (244, 220, 229),
                             (219, 229, 226), (224, 215, 233)))
    # 字体颜色，默认为蓝色
    fontcolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    image = Image.new('RGB',(width,height),bgcolor) #创建图片,RGB生成3通道
    font = ImageFont.truetype(font_path,53) #验证码的字体
    draw = ImageDraw.Draw(image)  #创建画笔

    text = random_captcha_text()  # 生成4个字符串
    text = ''.join(text)  # 将4个字符串合在一起变成一个字符串
    #text = gene_text() #生成字符串

    font_width, font_height = font.getsize(text)
    draw.text(((width - font_width) / num, (height - font_height) / num-7),text,\
            font= font,fill=fontcolor) #填充字符串
    if draw_line:
        gene_line(draw,width,height)
    #image = image.transform((width+30,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
    # image = image.transform((width+20,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
    #image = image.filter(ImageFilter.EDGE_ENHANCE_MORE) #滤镜，边界加强
    # aa = str(".png")
    # path = filename + text + aa
    # # # cv2.imwrite(path, I1)
    # # # image.save('idencode.jpg') #保存验证码图片
    # with open(path, "wb") as f:
    #      image.save(f)
    #image.save(path)
    # image = Image.open(path)#读取图片
    #image = image.convert("RGB") #RGBA生成的图像是4通道，转化成3通道
    captcha_image = np.array(image)#将对象转化为图像矩阵
    return text, captcha_image


# text,image=gen_captcha_text_and_image()
# plt.imshow(image)#imshow()对图像进行绘制,参数 image 要绘制的图像或数组
# plt.show()#调用show()函数来进行显示


# x=1
# # if __name__ == "__main__":
# # for k in(1,1000):
# while x<20:
#      gen_captcha_text_and_image()
#      x+=1


if __name__ == '__main__':
    # 测试
    text, image = gen_captcha_text_and_image()

    f = plt.figure()#创建一个窗口
    ax = f.add_subplot(111)#将画布分割成1行1列，图像画在从左到右从上到下的第1块
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)#坐标 ：x=0.1,y=0.9 ;字符串文本 ：text
    plt.imshow(image)#imshow()对图像进行绘制,参数 image 要绘制的图像或数组

    #plt.show()#调用show()函数来进行显示


# x=1
# # if __name__ == "__main__":
# # for k in(1,1000):
# while x<20:
#      gen_captcha_text_and_image()
#      x+=1
