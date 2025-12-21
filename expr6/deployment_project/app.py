# -------------------- app.py --------------------
import os
import base64
import datetime  # 引入 datetime 用于生成时间戳
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
# 假设 model_inference.py 文件存在并包含 load_models 和 predict_image 函数
from model_inference import load_models, predict_image

# --- 1. 配置 ---
app = Flask(__name__)
# 设置文件上传目录
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 允许的图片文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- 2. 状态存储 (历史记录) ---
# 使用全局列表存储预测历史，应用关闭后数据会丢失
prediction_history = []

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 3. 模型初始化 (服务器启动时加载一次) ---
# 注意: 模型路径使用用户的相对路径设置
AE_MODEL_PATH = '../autoencoder_model.pth'
CNN_MODEL_PATH = '../cnn_model.pth'

# 尝试加载模型
ae_model, cnn_model = load_models(AE_MODEL_PATH, CNN_MODEL_PATH)
if ae_model is None or cnn_model is None:
    print("FATAL ERROR: Models could not be loaded. Please check model files and structures.")
    # 如果模型加载失败，应用仍可运行，但预测功能将报错


def allowed_file(filename):
    """检查文件后缀名是否在允许列表中"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    """主页：显示上传表单和历史记录"""
    # 第一次访问或 GET 请求时，显示历史记录
    return render_template('index.html', history=prediction_history)


@app.route('/predict', methods=['POST'])
def upload_file():
    """处理图片上传和预测请求"""

    global prediction_history  # 声明使用全局历史列表

    # 1. 检查文件上传
    if 'file' not in request.files:
        return render_template('index.html', result="请求中缺少文件部分。", error=True, history=prediction_history)

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', result="未选择任何文件。", error=True, history=prediction_history)

    if not allowed_file(file.filename):
        return render_template('index.html', result="不允许的文件类型。请使用 PNG, JPG 或 JPEG。", error=True,
                               history=prediction_history)

    if ae_model is None or cnn_model is None:
        return render_template('index.html', result="服务器错误：模型未成功加载。", error=True,
                               history=prediction_history)

    # 2. 保存文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 3. 读取图片并编码为 Base64 (用于前端显示)
    b64_img = None
    try:
        with open(filepath, "rb") as image_file:
            b64_bytes = base64.b64encode(image_file.read())
            b64_string = b64_bytes.decode('utf-8')
            mime_type = f"image/{filename.rsplit('.', 1)[1].lower()}"
            b64_img = f"data:{mime_type};base64,{b64_string}"
    except Exception as e:
        print(f"Error reading image for Base64 encoding: {e}")

    # 4. 调用核心预测函数
    predicted_class, confidence = predict_image(filepath, ae_model, cnn_model)

    # 5. 记录历史 (已新增 image_base64 字段)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        'timestamp': timestamp,
        'filename': filename,
        'prediction': predicted_class,
        'confidence': f"{confidence:.2f}",
        'image_base64': b64_img # <--- 关键：将 Base64 数据存入历史记录
    }
    prediction_history.insert(0, record)  # 将新记录插入列表头部

    # 6. 删除临时保存的文件
    os.remove(filepath)

    # 7. 返回结果
    result_text = f"预测结果: {predicted_class} (置信度: {confidence:.2f})"
    return render_template('index.html',
                           result=result_text,
                           prediction=predicted_class,
                           uploaded_image=b64_img,  # Base64 数据用于显示图片
                           history=prediction_history)


if __name__ == '__main__':
    # 启动 Flask 应用
    app.run(debug=True, port=5000)