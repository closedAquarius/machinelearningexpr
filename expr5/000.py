import os
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import gc

# 统一 Keras 导入路径，使用 TensorFlow 标准 API
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import InputLayer, Input, Reshape, MaxPooling2D, Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# ====================================================================
# 0. 实验配置与输出设置
# ====================================================================

# 设置图片输出文件夹路径
OUTPUT_DIR = 'experiment_outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"所有生成的图片将保存到目录: {OUTPUT_DIR}")

# --- 中文字体设置 ---
#    检查系统中是否存在 SimHei 字体 (最常见的中文支持字体)
#    如果 SimHei 不存在，可能需要手动安装，或改为使用其他常见字体如 'Microsoft YaHei'
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
except ValueError:
    print("SimHei 字体未找到，尝试使用系统默认字体...")
    # 备用方案：使用系统默认的 sans-serif 字体
    pass

plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题
# --- 中文字体设置结束 ---

# ====================================================================
# 1. 准备工作及辅助函数定义 (包含图像保存逻辑)
# ====================================================================

# 1.2 载入数据 (使用用户指定的 Keras/TensorFlow 方式)
print("\n--- 1.2 载入数据 ---")
# MNIST 数据集 (x_train/x_test 形状: (n, 28, 28), y_train/y_test 形状: (n,))
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1.3 配置神经网络参数
print("\n--- 1.3 配置神经网络参数 ---")
img_size = x_train.shape[1]  # 28
img_size_flat = img_size * img_size  # 784
img_shape = (img_size, img_size)  # (28, 28)
img_shape_full = (img_size, img_size, 1)  # (28, 28, 1)
num_classes = 10  # 10
num_channels = 1  # 1

print(f"img_size: {img_size}")
print(f"img_size_flat: {img_size_flat}")
print(f"img_shape: {img_shape}")
print(f"img_shape_full: {img_shape_full}")
print(f"num_classes: {num_classes}")
print(f"num_channels: {num_channels}")


# 1.4 绘制图像的辅助函数 (修改为保存图片)
def plot_images(images, cls_true, cls_pred=None, filename="images_plot.png"):
    """绘制9张图像，并显示它们的真实类别和可选的预测类别。"""
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = f"True: {cls_true[i]}"
        else:
            xlabel = f"True: {cls_true[i]}, Pred: {cls_pred[i]}"
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # 保存图像
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)  # 关闭图形以节省内存
    print(f"图片已保存: {filename}")


# 1.5 绘制错误分类图像的辅助函数 (依赖 plot_images)
def plot_example_errors(cls_pred, correct, filename="error_examples.png"):
    """绘制测试集中分类错误的图像样本。"""
    incorrect = (correct == False)
    images = x_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = y_test[incorrect]

    # 确保至少有9张错误图片
    num_to_plot = min(9, len(images))
    if num_to_plot == 0:
        print("未找到错误分类的图像.")
        return

    # 需要手动创建一个只有 num_to_plot 张图片的 figure
    fig, axes = plt.subplots(1, num_to_plot)
    if num_to_plot == 1:
        axes = [axes]  # 确保 axes 是一个可迭代对象
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        xlabel = f"True: {cls_true[i]}, Pred: {cls_pred[i]}"
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # 保存图像
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"图片已保存: {filename}")


# 5.1 画卷积权重的辅助函数 (修改为保存图片)
def plot_conv_weights(weights, input_channel=0, filename="conv_weights.png"):
    w_min = np.min(weights)
    w_max = np.max(weights)
    num_filters = weights.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = weights[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"图片已保存: {filename}")


# 5.4 绘制卷积层输出的帮助函数 (修改为保存图片)
def plot_conv_output(values, filename="conv_output.png"):
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"图片已保存: {filename}")


# 5.5 定义单个图像绘制函数 (修改为保存图片)
def plot_image(image, filename="single_image.png"):
    fig, ax = plt.subplots()
    ax.imshow(image.reshape(img_shape),
              interpolation='nearest',
              cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"图片已保存: {filename}")


# ---------------------------------------------
# 1.4 检测函数: 绘制测试集前9张原始图片
# ---------------------------------------------
images_test_9 = x_test[0:9]
cls_true_test_9 = y_test[0:9]
plot_images(images=images_test_9, cls_true=cls_true_test_9, filename="1_4_test_set_9_images.png")

# ====================================================================
# 预处理数据 (用于模型训练和评估)
# ====================================================================

print("\n--- 数据预处理 ---")
# 1. 归一化：0-255 -> 0.0-1.0
x_train_normalized = x_train.astype('float32') / 255.0
x_test_normalized = x_test.astype('float32') / 255.0

# 2. 扁平化：(n, 28, 28) -> (n, 784)
x_train_flat = x_train_normalized.reshape(x_train_normalized.shape[0], img_size_flat)
x_test_flat = x_test_normalized.reshape(x_test_normalized.shape[0], img_size_flat)

# 3. 标签 One-Hot 编码：(n,) -> (n, 10)
y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

print(f"x_train_flat 形状: {x_train_flat.shape}")
print(f"y_train_one_hot 形状: {y_train_one_hot.shape}")

# ====================================================================
# 2. 序列模型 (Sequential Model)
# ====================================================================

print("\n--- 2. 序列模型 ---")

# 2.1 模型框架
model = Sequential([
    InputLayer(input_shape=(img_size_flat,)),
    Reshape(img_shape_full),

    Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'),
    MaxPooling2D(pool_size=2, strides=2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
print("2.1 序列模型架构定义完成.")
model.summary()

# 2.2 模型编译
optimizer_seq = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer_seq,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("2.2 序列模型编译完成.")

# 2.3 训练
print("2.3 开始序列模型训练...")
model.fit(x=x_train_flat,
          y=y_train_one_hot,
          epochs=1,
          batch_size=128)
print("序列模型训练完成.")

# 2.4 评估与性能指标
print("\n2.4 评估序列模型性能...")
result_seq = model.evaluate(x=x_test_flat, y=y_test_one_hot, verbose=1)
print("\n--- 序列模型性能指标 ---")
for name, value in zip(model.metrics_names, result_seq):
    print(f"{name}: {value}")

# 2.5 预测前九张图片
print("\n2.5 序列模型预测前9张图片...")
y_pred_seq_9 = model.predict(x=x_test_flat[0:9])
cls_pred_seq_9 = np.argmax(y_pred_seq_9, axis=1)
plot_images(images=images_test_9,
            cls_true=cls_true_test_9,
            cls_pred=cls_pred_seq_9,
            filename="2_5_sequential_model_prediction.png")

# 2.6 错分类的图片
print("\n2.6 绘制序列模型错分类的图片...")
y_pred_seq_all = model.predict(x=x_test_flat)
cls_pred_seq_all = np.argmax(y_pred_seq_all, axis=1)
correct_seq = (cls_pred_seq_all == y_test)
plot_example_errors(cls_pred=cls_pred_seq_all,
                    correct=correct_seq,
                    filename="2_6_sequential_model_errors.png")

# ====================================================================
# 3. 功能模型 (Functional Model)
# ====================================================================

print("\n--- 3. 功能模型 ---")

# 3.1 模型框架
inputs = Input(shape=(img_size_flat,))
net = inputs
net = Reshape(img_shape_full)(net)
net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1_func')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2_func')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Flatten()(net)
net = Dense(128, activation='relu')(net)
net = Dense(num_classes, activation='softmax')(net)
outputs = net

# 创建模型实例
model2 = Model(inputs=inputs, outputs=outputs)
print("3.1 功能模型架构定义完成.")
model2.summary()

# 3.2 模型编译
model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
print("3.2 功能模型编译完成.")

# 3.3 训练
print("3.3 开始功能模型训练...")
model2.fit(x=x_train_flat,
           y=y_train_one_hot,
           epochs=1,
           batch_size=128)
print("功能模型训练完成.")

# 3.4 评估
print("\n3.4 评估功能模型性能...")
result_func = model2.evaluate(x=x_test_flat, y=y_test_one_hot, verbose=1)
print("\n--- 功能模型性能指标 ---")
for name, value in zip(model2.metrics_names, result_func):
    print(f"{name}: {value}")

accuracy_index_func = model2.metrics_names.index('accuracy')
print(f"准确率 (Accuracy): {result_func[accuracy_index_func]}")

# 3.5 错分类的图片 (参照 2.5 和 2.6)
print("\n3.5 绘制功能模型错分类的图片...")
y_pred_func_all = model2.predict(x=x_test_flat)
cls_pred_func_all = np.argmax(y_pred_func_all, axis=1)
correct_func = (cls_pred_func_all == y_test)
plot_example_errors(cls_pred=cls_pred_func_all,
                    correct=correct_func,
                    filename="3_5_functional_model_errors.png")

# ====================================================================
# 4. 保存和加载模型
# ====================================================================

print("\n--- 4. 保存和加载模型 ---")

# 4.1 保存 Keras 模型
path_model = os.path.join(OUTPUT_DIR, 'functional_model.keras')
model2.save(path_model)
print(f"4.1 模型 model2 已成功保存到: {path_model}")

# 4.2 删除模型
del model2
gc.collect()
print("4.2 模型 model2 已从内存中删除.")

# 4.3 加载模型
model3 = load_model(path_model)
print("4.3 模型 model3 加载成功.")

# 4.4 用加载的模型来预测
print("4.4 使用加载的模型 model3 预测前9张图片...")
y_pred_model3 = model3.predict(x=x_test_flat[0:9])
cls_pred_model3 = np.argmax(y_pred_model3, axis=1)
plot_images(images=images_test_9,
            cls_true=cls_true_test_9,
            cls_pred=cls_pred_model3,
            filename="4_4_loaded_model_prediction.png")

# ====================================================================
# 5. 权重和输出的可视化
# ====================================================================

print("\n--- 5. 权重和输出的可视化 ---")

# 5.2 得到层 (使用 model3)
model3.summary()
layer_input = model3.layers[0]
# 这里的索引是 Functional Model 的层，注意和 Sequential Model 的区别
layer_conv1 = model3.layers[2]  # Conv2D name='layer_conv1_func'
layer_conv2 = model3.layers[4]  # Conv2D name='layer_conv2_func'
print(f"5.2 获取的层名称: Conv1={layer_conv1.name}, Conv2={layer_conv2.name}")

# 5.3 卷积权重
weights_conv1 = layer_conv1.get_weights()[0]
plot_conv_weights(weights=weights_conv1, input_channel=0, filename="5_3_weights_conv1.png")

weights_conv2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights=weights_conv2, input_channel=0, filename="5_3_weights_conv2.png")

# 5.5 输入图像 (用于卷积层输出可视化)
image1 = x_test[0]  # 原始 (28, 28) 图像
image1_normalized = image1.astype('float32') / 255.0
image1_flat_batch = image1_normalized.reshape(1, img_size_flat)  # (1, 784) 扁平化并带批次维度

plot_image(image1, filename="5_5_input_image1.png")

# 5.6 卷积层输出之方法一 (基于 K 函数)
# 5.6.1 基于 K 函数的模型转换函数
output_conv1_func = K.function(inputs=[layer_input.input],
                               outputs=[layer_conv1.output])

# 5.6.2 获取卷积层1的输出
# 传入预处理好的带有批次维度的输入
layer_output1 = output_conv1_func([image1_flat_batch])[0]
print(f"5.6.2 layer_output1 的形状: {layer_output1.shape}")

# 5.6.3 绘制输出
plot_conv_output(values=layer_output1, filename="5_6_output_conv1.png")

# 5.7 卷积层输出之方法二 (基于 Model 函数)
# 5.7.1 基于 Model 函数的图层输出获取
output_conv2_model = Model(inputs=layer_input.input,
                           outputs=layer_conv2.output)

# 5.7.2 获取卷积层2的输出
# 使用 predict 方法
layer_output2 = output_conv2_model.predict(image1_flat_batch)
print(f"5.7.2 layer_output2 的形状: {layer_output2.shape}")

# 5.7.3 绘制输出
plot_conv_output(values=layer_output2, filename="5_7_output_conv2.png")

print("\n========== 实验脚本执行完毕 ==========")
print(f"所有生成的图片已保存在 ./{OUTPUT_DIR} 目录中。")