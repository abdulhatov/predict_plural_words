import label as label
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# Преобразование строковых меток в числовые
label_encoder = LabelEncoder()

from padej.source_multi import (
    texts,
    labels,
)
print(texts)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(sequences)

#label_mapping = {"дар": 0, "лар": 1, "лер": 2, "лор": 3,"лөр":4,  "тар": 5, "тор": 6, "тер":7, "төр" : 8, "дор": 9, 'дер':10, "дөр":11}
label_mapping = {'лар': 0, 'лер': 1, 'лор': 2, 'лөр': 3, 'дар': 4, 'дор': 5, 'дер':6, 'дөр' : 7, 'тар': 8, 'тор' : 9, 'тер': 10, 'төр': 11}
numeric_labels = np.array([label_mapping[label] for label in labels], dtype=np.int32)


label_mapping_inv = {v: k for k, v in label_mapping.items()}

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=12, input_length=padded_sequences.shape[1]))
model.add(LSTM(32))
model.add(Dense(12, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

# үйрөтүү

epochs = 20
history = model.fit(padded_sequences, numeric_labels, epochs=epochs, validation_split=0.30)

# тактыктын жана катанын маанисин алуу
train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
test_loss = history.history['val_loss']

epochs = range(1, len(train_accuracy) + 1)

# тактыктын графиги
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, 'bo-', label='Training accuracy')
plt.plot(epochs, test_accuracy, 'ro-', label='Testing accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# катанын графиги
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='Training loss')
plt.plot(epochs, test_loss, 'ro-', label='Testing loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# мисал
while True:
    input_text = list(map(str, input("Enter : ").split()))
    new_sequences = tokenizer.texts_to_sequences(input_text)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1])
    predictions = model.predict(new_padded_sequences)
    predicted_labels = [label_mapping_inv[tf.argmax(prediction).numpy()] for prediction in predictions]
    print(input_text[0]+predicted_labels[0])
