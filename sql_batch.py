import base64
import json
import tensorflow as tf
import numpy as np
import random
import os
import sqlite3  # 使用 SQLite 而不是 MySQL
from cnnlib.network import CNN
from flask import Flask, request, jsonify, Response
from PIL import Image
from io import BytesIO
from cnnlib.recognition_object import Recognizer

# 设置 TensorFlow 使用 CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 读取配置参数
with open("conf/sample_config.json", "r") as f:
    sample_conf = json.load(f)

# 配置参数
image_height = sample_conf["image_height"]
image_width = sample_conf["image_width"]
max_captcha = sample_conf["max_captcha"]
api_image_dir = sample_conf["api_image_dir"]
model_save_dir = sample_conf["model_save_dir"]
image_suffix = sample_conf["image_suffix"]  # 文件后缀
use_labels_json_file = sample_conf['use_labels_json_file']

# 加载字符集
if use_labels_json_file:
    with open("tools/labels.json", "r") as f:
        char_set = f.read().strip()
else:
    char_set = sample_conf["char_set"]

# 创建 Flask 应用实例
app = Flask(__name__)

R = Recognizer(image_height, image_width, max_captcha, char_set, model_save_dir)  # 创建识别器实例

def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

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
            test_image = self.convert2gray(test_image)  # 转换图像为灰度
            test_image = test_image.flatten() / 255  # 扁平化并归一化
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
            text_list = sess.run(predict, feed_dict={self.X: [test_image], self.keep_prob: 1.})
            predict_text = []
            for p in text_list[0].tolist():
                predict_text.append(self.char_set[p])  # 添加预测字符
            predict_text = "".join(predict_text)  # 将字符连接成字符串
            return predict_text

def some_validation_function(result, predict_text):
    # 检查预测文本是否与预期结果匹配
    return result == predict_text

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json  # 从请求中获取 JSON 数据
    base64_image = data.get('image')  # 提取 Base64 图像字符串

    tb = TestBatch(char_set, model_save_dir)  # 实例化 TestBatch
    image = tb.convert_base64_to_image(base64_image)  # 转换图像

    result = tb.test_batch(image)  # 识别验证码

    # 连接到 SQLite 数据库
    connection = sqlite3.connect('captcha_requests.db')
    try:
        with connection:
            # 如果不存在则创建表
            connection.execute(
                "CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY AUTOINCREMENT, result TEXT, success BOOLEAN)"
            )
            is_successful = some_validation_function(result, '')  # 验证结果
            # 插入识别结果和成功状态
            connection.execute("INSERT INTO requests (result, success) VALUES (?, ?)", (result, is_successful))
    finally:
        connection.close()

    return jsonify({'result': result})

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello World!"

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=7000,
        debug=True
    )
