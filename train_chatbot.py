import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Пример текстов для обучения
texts = [
    "Привет!",
    "Как дела?",
    "Что нового?",
    "Как твои дела?",
    "Пока!",
    "До свидания!",
]

# Ответы на соответствующие вопросы
responses = [
    "Привет!",
    "Все отлично, спасибо!",
    "Ничего особенного. А у тебя?",
    "У меня все хорошо, спасибо за спрос!",
    "Пока! Возвращайся еще!",
    "До свидания! Приходи еще!",
]

# Инициализация токенизатора
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Преобразование текстов в последовательности чисел
sequences = tokenizer.texts_to_sequences(texts)

# Определение максимальной длины последовательности
max_len = max(len(seq) for seq in sequences)

# Приведение всех последовательностей к одной длине
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Создание модели
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_len),
    LSTM(16),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(padded_sequences, np.array(tf.argmax(tf.one_hot(tokenizer.texts_to_sequences(responses), depth=len(tokenizer.word_index) + 1), axis=2)), epochs=100)

# Сохранение модели
model.save('chatbot_model.h5')

