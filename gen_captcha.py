from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']



#生成四个字符， 验证码一般都无视大小写；验证码长度4个字符
#返回一个含有四个字符串的列表
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)#choice()方法返回一个列表，元组或字符串的随机项
        captcha_text.append(c)
    return captcha_text



# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()#生成4个字符串
    captcha_text = ''.join(captcha_text)#将4个字符串合在一起变成一个字符串

    captcha = image.generate(captcha_text) #生成验证码
    #image.write(captcha_text,'captcha/images/' captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)#读取图片
    captcha_image = np.array(captcha_image)#将对象转化为图像矩阵
    return captcha_text, captcha_image



if __name__ == '__main__':
    # 测试
    text, image = gen_captcha_text_and_image()

    f = plt.figure()#创建一个窗口
    ax = f.add_subplot(111)#将画布分割成1行1列，图像画在从左到右从上到下的第1块
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)#坐标 ：x=0.1,y=0.9 ;字符串文本 ：text
    plt.imshow(image)#imshow()对图像进行绘制,参数 image 要绘制的图像或数组

    plt.show()#调用show()函数来进行显示
