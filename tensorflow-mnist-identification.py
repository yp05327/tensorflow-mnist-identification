# coding=utf-8 
import tensorflow as tf
import numpy as np
import os
import sys
import argparse

from PIL import Image, ImageFilter

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

def analysis(im):
    # 读取保存的模型
    images_placeholder=tf.placeholder(tf.float32, shape=(1,mnist.IMAGE_PIXELS))
    logits = mnist.inference(images_placeholder,128,32)
    init_op = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.path.abspath('.')+'/model.ckpt-49999')
        prediction=tf.argmax(logits,1)
        return prediction.eval(feed_dict={images_placeholder: [im]}, session=sess)
    
def change_to_rgb(img):
    img.load()
    newimg = Image.new('RGB', img.size, (255, 255, 255))
    newimg.paste(img, mask=img.split()[3])
    return newimg

def get_image(file):
    # 读取图片
    im = Image.open(file)
    
    if im.mode == 'RGBA':
        im = change_to_rgb(im)
    
    width = im.size[0]
    height = im.size[1]
    pix = im.load()
    # 获取切割框
    box = [width/2,height/2,width/2,height/2]
    for x in range(width):
        for y in range(height):
            r,g,b = pix[x,y]
            if [r,g,b] != [255,255,255]:
                if x < box[0]:
                    box[0] = x
                if x > box[2]:
                    box[2] = x
                if y < box[1]:
                    box[1] = y
                if y > box[3]:
                    box[3] = y
    # 切割
    im = im.crop(box)
    width = im.size[0]
    height = im.size[1] 
    # 生成标准图像
    base_scale = 0.2
    base_width = int(width * (1 + base_scale))
    base_height = int(height * (1 + base_scale))
    baseImage = Image.new('RGB', (base_width, base_height), (255,255,255))
    baseImage.paste(im, ((base_width - width)/2, (base_height - height)/2))
    
    # 将图像数据归一化并转换成list
    baseImage.thumbnail((28,28))
    baseImage=baseImage.convert('L')
    width = float(baseImage.size[0])
    height = float(baseImage.size[1])
    
    ''' 以下代码来源于  The following code comes from:
        https://github.com/niektemme/tensorflow-mnist-predict/blob/master/predict_2.py
        作者：niektemme  Author:niektemme
        原始代码有一些拼写错误，我已经修正
        The original code have some spelling mistake, and i have corrected them.
    '''
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1  
        # resize and sharpen
        img = baseImage.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = baseImage.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    
    tv = list(newImage.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    tva = np.array(tva)
    return tva

def main(_):
    im = get_image(FLAGS.image_dir)
    result = analysis(im)
    if FLAGS.del_image:
        os.remove(FLAGS.image_dir)
    print str(result[0])
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='num.jpg',
      help='Directory of the image'
  )
  parser.add_argument(
      '--del_image',
      default=False,
      help='Whether delete the image'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
