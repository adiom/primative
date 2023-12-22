import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка обученной модели
model = load_model('chatbot_model.h5')

# Загрузка токенизатора
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["Привет!", "Как дела?", "Что нового?", "Как твои дела?", "Пока!", "До свидания!"])

# Тестирование с вопросом пользователя
user_input = "Привет, как дела?"
user_sequence = tokenizer.texts_to_sequences([user_input])
user_padded_sequence = pad_sequences(user_sequence, maxlen=len(tokenizer.word_index) + 1, padding='post')

# Получение ответа от модели
prediction = model.predict(user_padded_sequence)
predicted_index = np.argmax(prediction)

# Вывод ответа
print("Ответ чатбота:", tokenizer.index_word[predicted_index])

