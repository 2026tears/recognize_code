import json
import tensorflow as tf
import numpy as np
import time
import base64
import random
import os
import pymysql  # 导入 pymysql
from cnnlib.network import CNN
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import io

from cnnlib.recognition_object import Recognizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    #这两行代码将设置 TensorFlow 使用 CPU 而不是 GPU，适用于未安装 GPU 的环境或调试时的情况

with open("conf/sample_config.json", "r") as f:
    sample_conf = json.load(f)       #读取配置文件 sample_config.json 并将其解析为 Python 字典 sample_conf

image_height = sample_conf["image_height"]
image_width = sample_conf["image_width"]
max_captcha = sample_conf["max_captcha"]
api_image_dir = sample_conf["api_image_dir"]
model_save_dir = sample_conf["model_save_dir"]
image_suffix = sample_conf["image_suffix"]  # 文件后缀
use_labels_json_file = sample_conf['use_labels_json_file']

if use_labels_json_file:
    with open("tools/labels.json", "r") as f:
        char_set = f.read().strip()
else:
    char_set = sample_conf["char_set"]     #从配置文件中提取了多个重要参数，包括图像的高度、宽度、最大验证码字符数量和模型保存目录等
                                            # 根据 use_labels_json_file 选择读取字符集，决定是否从 labels.json 文件中提取字符集，也可以直接从配置文件中读取。

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))    #创建一个 Flask 应用实例 app
                                                        #basedir 用于获取当前文件目录的绝对路径

R = Recognizer(image_height, image_width, max_captcha, char_set, model_save_dir)  #生成一个验证码识别对象 R，用于后续的验证码识别操作

def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp                                          #定义一个函数 response_headers，用于生成带有跨域访问控制的响应 这是为了允许前端应用从不同域访问这个 API



class TestError(Exception):
    pass


class TestBatch(CNN):
    def __init__(self, char_set, model_save_dir):
        self.model_save_dir = model_save_dir
        self.channel = 1
        self.max_captcha = 4
        self.char_set = char_set
        self.char_set_len = len(char_set)

        # 初始化模型
        super(TestBatch, self).__init__(60, 160, self.max_captcha, char_set, model_save_dir)

    def convert_base64_to_image(self, base64_str):
        image_data = base64.b64decode(base64_str)  #将 Base64 字符串转换为图像
        bytes_io = BytesIO(image_data)
        image = Image.open(bytes_io)
        return np.array(image)

    def test_batch(self, test_image):
        y_predict = self.model()
        right = 0

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_dir)
            test_image = self.convert2gray(test_image) #将验证码图像转换为灰度图像
            test_image = test_image.flatten() / 255 #将二维图像展平成一维数组，并归一化到[0, 1]范围（通过除以255）
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
            text_list = sess.run(predict, feed_dict={self.X: [test_image], self.keep_prob: 1.})
            predict_text = []    #创建一个空列表 predict_text 用于存储预测的字符
            for p in text_list[0].tolist():
                predict_text.append(self.char_set[p]) #遍历 text_list[0].tolist() 中的每个索引 p，将其对应的字符（通过 self.char_set[p]）添加到 predict_text 列表中
                predict_text = "".join(predict_text)  #使用 "".join(predict_text) 将列表中的字符合并成一个字符串
                return predict_text


def some_validation_function(result, predict_text):
    # 检查识别结果与期望的正确结果是否一致
    return result == predict_text


@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json  # 获取请求的 JSON 数据
    base64_image = data.get('image')  # 提取 Base64 图像字符串

    tb = TestBatch(char_set, model_save_dir) #实例化
    image = tb.convert_base64_to_image(base64_image)  # 转换图像

    result = tb.test_batch(image)  # 识别验证码

    # 连接到 MySQL 数据库
    connection = pymysql.connect(
        host='localhost',       # 你的 MySQL 主机
        port=3306,            #你的端口号
        user='root',   # 你的 MySQL 用户名
        password='123456', # 你的 MySQL 密码
        db='recoginize_code'
    )
    try:
                with connection.cursor() as cursor:
                    # 创建表并新增 success 列
                    cursor.execute(
                        "CREATE TABLE IF NOT EXISTS requests (id INT PRIMARY KEY AUTO_INCREMENT, result TEXT, success BOOLEAN)")
                    # 假设通过某种方法判断识别结果是否成功
                    is_successful = some_validation_function(result)  # 判断是否正确
                    # 将识别结果与成功状态一起插入
                    cursor.execute("INSERT INTO requests (result, success) VALUES (%s, %s)", (result, is_successful))
                    connection.commit()
    finally:
                connection.close()

    return jsonify({'result': result})


if __name__ == '__main__':
    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    model_save_dir = sample_conf["model_save_dir"]
    use_labels_json_file = sample_conf['use_labels_json_file']

    if use_labels_json_file:
        with open("tools/labels.json", "r") as f:
            char_set = f.read().strip()
    else:
        char_set = sample_conf["char_set"]

    app.run(
        host='0.0.0.0',
        port=7000,
        debug=True
    )

