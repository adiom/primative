import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Задаем данные для обучения
training_data = [
    "Привет",
    "Как дела?",
    "Что нового?",
    "Пока",
]

training_responses = [
    "Привет!",
    "Отлично!",
    "Ничего особенного.",
    "До свидания!",
]

# Создаем токенизатор
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_data + training_responses)

# Количество слов в словаре
vocab_size = len(tokenizer.word_index) + 1

# Преобразуем обучающие данные в последовательности чисел
sequences = tokenizer.texts_to_sequences(training_data)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Преобразуем ответы в последовательности чисел
responses_sequences = tokenizer.texts_to_sequences(training_responses)
padded_responses = pad_sequences(responses_sequences, maxlen=max_len, padding='post')

# Определяем модель
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, 16, input_length=max_len),
    keras.layers.GRU(32),
    keras.layers.Dense(vocab_size, activation='softmax')
])

# Компилируем модель
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучаем модель
model.fit(padded_sequences, np.array(padded_responses), epochs=100)

# Создаем функцию для генерации ответов
def generate_response(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post')
    predicted_output = model.predict(input_sequence)[0]
    predicted_index = np.argmax(predicted_output)
    return tokenizer.index_word[predicted_index]

# Тестируем модель
test_input = "Привет"
predicted_response = generate_response(test_input)
print(f'Вход: {test_input}')
print(f'Ответ: {predicted_response}')

