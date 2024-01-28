'''
Author: callus
Date: 2024-01-25 22:34:02
LastEditors: callus
Description: some description
FilePath: /drug-shortage-forecast/2024year/LSTMdemo.py
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 假设你的数据已经预处理好，并且分割为特征和标签
# X_train, y_train = ... # 训练数据的特征和标签
# X_test, y_test = ... # 测试数据的特征和标签

# LSTM模型的参数
input_shape = (X_train.shape[1], X_train.shape[2]) # 根据你的数据调整
num_units = 50 # LSTM层的单元数
dropout_rate = 0.2 # Dropout层的比率
num_classes = len(np.unique(y_train)) # 分类任务的类别数

# 构建模型
model = Sequential()
model.add(LSTM(units=num_units, input_shape=input_shape, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(LSTM(units=num_units))
model.add(Dropout(dropout_rate))
model.add(Dense(units=num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
