# from gen_captcha import gen_captcha_text_and_image
# from gen_captcha import number
# from gen_captcha import alphabet
# from gen_captcha import ALPHABET

from custom import gen_captcha_text_and_image
from custom import number
from custom import alphabet
from custom import ALPHABET

import time
import numpy as np
import tensorflow as tf

text, image = gen_captcha_text_and_image()
print("verification code iamge channel:", image.shape)  # (60, 160, 3)


# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)
print("Max number of label:", MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


####################################################################
# 占位符，X和Y分别是输入训练数据和其标签，标签转换成8*10的向量
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
# 声明dropout占位符变量
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    # 把 X reshape 成 IMAGE_HEIGHT*IMAGE_WIDTH*1的格式,输入的是灰度图片，所有通道数是1;
    # shape 里的-1表示数量不定，根据实际情况获取，这里为每轮迭代输入的图像数量（batchsize）的大小;
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH))
    # w_c2_alpha = np.sqrt(2.0/(3*3*32)) 
    # w_c3_alpha = np.sqrt(2.0/(3*3*64)) 
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    # 搭建第一层卷积层
    # shape[3, 3, 1, 32]里前两个参数表示卷积核尺寸大小，即patch;
    # 第三个参数是图像通道数，第四个参数是该层卷积核的数量，有多少个卷积核就会输出多少个卷积特征图像
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    # 每个卷积核都配置一个偏置量，该层有多少个输出，就应该配置多少个偏置量
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    # 图片和卷积核卷积，并加上偏执量，卷积结果28x28x32
    # tf.nn.conv2d() 函数实现卷积操作
    # tf.nn.conv2d()中的padding用于设置卷积操作对边缘像素的处理方式，在tf中有VALID和SAME两种模式
    # padding='SAME'会对图像边缘补0,完成图像上所有像素（特别是边缘象素）的卷积操作
    # padding='VALID'会直接丢弃掉图像边缘上不够卷积的像素
    # strides：卷积时在图像每一维的步长，是一个一维的向量，长度4，并且strides[0]=strides[3]=1
    # tf.nn.bias_add() 函数的作用是将偏置项b_c1加到卷积结果value上去;
    # 注意这里的偏置项b_c1必须是一维的，并且数量一定要与卷积结果value最后一维数量相同
    # tf.nn.relu() 函数是relu激活函数，实现输出结果的非线性转换，即features=max(features, 0)，输出tensor的形状和输入一致
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    # tf.nn.max_pool()函数实现最大池化操作，进一步提取图像的抽象特征，并且降低特征维度
    # ksize=[1, 2, 2, 1]定义最大池化操作的核尺寸为2*2, 池化结果14x14x32 卷积结果乘以池化卷积核
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # tf.nn.dropout是tf里为了防止或减轻过拟合而使用的函数，一般用在全连接层;
    # Dropout机制就是在不同的训练过程中根据一定概率（大小可以设置，一般情况下训练推荐0.5）随机扔掉（屏蔽）一部分神经元，
    # 不参与本次神经网络迭代的计算（优化）过程，权重保留但不做更新;
    # tf.nn.dropout()中 keep_prob用于设置概率，需要是一个占位变量，在执行的时候具体给定数值
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过神经网络第一层卷积（图像尺寸不变、特征×32）、池化（图像尺寸缩小一半，特征不变）之后;
    # 输出大小为 30*80*32

    # 搭建第二层卷积层
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过神经网络第一层后输出大小为 30*80*32
    # 经过神经网络第二层运算后输出为 16*40*64 (30*80的图像经过2*2的卷积核池化，padding为SAME，输出维度是16*40)

    # 搭建第三层卷积层
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过神经网络第一层后输出大小为 30*80*32 经过第二层后输出为 16*40*64
    # 经过神经网络第二层运算后输出为 16*40*64 ; 经过第三层输出为 8*20*64，这个参数很重要，决定量后边全连接层的维度

    # 搭建全连接层
    # 二维张量，第一个参数8*20*64的patch，这个参数由最后一层卷积层的输出决定，第二个参数代表卷积个数共1024个，即输出为1024个特征
    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    # 偏置项为1维，个数跟卷积核个数保持一致
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    # w_d.get_shape()作用是把张量w_d的形状转换为元组tuple的形式，w_d.get_shape().as_list()是把w_d转为元组再转为list形式
    # w_d 的 形状是[ 8 * 20 * 64, 1024]，w_d.get_shape().as_list()结果为 8*20*64=10240 ;
    # 所以tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])的作用是把最后一层隐藏层的输出转换成一维的形式
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    # tf.matmul(dense, w_d)函数是矩阵相乘，输出维度是 -1*1024
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    # 经过全连接层之后，输出为 一维，1024个向量

    # w_out定义成一个形状为 [1024, 8 * 10] = [1024, 80]
    with tf.name_scope('w_out'):
        w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))

    with tf.name_scope('b_out'):
        b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    # out 的输出为 8*10 的向量， 8代表识别结果的位数，10是每一位上可能的结果（0到9）
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    # 输出神经网络在当前参数下的预测值
    return out


# 训练
def train_crack_captcha_cnn():
    # with tf.device('/cpu:0'):
    output = crack_captcha_cnn()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    # tf.nn.sigmoid_cross_entropy_with_logits()函数计算交叉熵,输出的是一个向量而不是数;
    # 交叉熵刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近
    # tf.reduce_mean()函数求矩阵的均值
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
        tf.summary.scalar('loss', loss)  # 可视化loss常量
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    # tf.train.AdamOptimizer（）函数实现了Adam算法的优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/", sess.graph)

        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print("step is %s , loss is %g" % (step, loss_))

            # writer.add_summary(summary,step)
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                summary, acc = sess.run([merged, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("----------step is %s , acc is %g--------" % (step, acc))

                writer.add_summary(summary, step)

                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.50 :
                    saver.save(sess, "F://crack_capcha_model/crack_capcha.model", global_step=step)
                    break

            step += 1
    # performance test
    # if step == 20:
    #	break


if __name__ == '__main__':
    start = time.clock()

    train_crack_captcha_cnn()

    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
