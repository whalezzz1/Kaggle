#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/8 17:48
# @Author : ZhangJinggang
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

# Build a simple neural network model
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128)

# Evaluate the model
score = model.evaluate(x_test, y_test, batch_size=128)
print("Test accuracy: ", score[1])
