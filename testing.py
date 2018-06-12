#from gen_captcha import gen_captcha_text_and_image
from custom import gen_captcha_text_and_image

import matplotlib.pyplot as plt
from PIL import Image

from training import convert2gray
from training import vec2text
from training import crack_captcha_cnn

import time
import training as tr
import numpy as np
import tensorflow as tf



def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("F://crack_capcha_model"))

        predict = tf.argmax(tf.reshape(output, [-1, tr.MAX_CAPTCHA, tr.CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={tr.X: [captcha_image], tr.keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(tr.MAX_CAPTCHA * tr.CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * tr.CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)


# if __name__ == '__main__':
#     start = time.clock()
#
#     text, image = gen_captcha_text_and_image()
#
#     f = plt.figure() #创建一个窗口
#     ax = f.add_subplot(111)  #将画布分割成1行1列，图像画在从左到右从上到下的第1块
#     ax.text(0.1, 1.1, text, ha='center', va='center', transform=ax.transAxes)  #坐标 ：x=0.1,y=1.1 ;字符串文本 ：text
#     plt.imshow(image) #imshow()对图像进行绘制,参数 image 要绘制的图像或数组
#
#     image = convert2gray(image) # 把彩色图像转为灰度图像
#     image = image.flatten() / 255
#     predict_text = crack_captcha(image)
#     print("correct: {}  predict: {}".format(text, predict_text))
#
#     end = time.clock()
#     print('Running time: %s Seconds' % (end - start))

    #plt.show() #调用show()函数来进行显示




#预测单个图片
if __name__ == '__main__':

    qq = Image.open(r'F:/frame13856.png')
    captcha_image1 = np.array(qq)
    plt.imshow(captcha_image1)
    image1 = convert2gray(captcha_image1)
    image1 = np.pad(image1, ((5, 5), (15, 15)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
    image1 = image1.flatten() / 255
    predict_text1 = crack_captcha(image1)
    print('prodict:',predict_text1)
    #plt.show()
