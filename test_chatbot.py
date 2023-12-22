import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка обученной модели
model = tf.keras.models.load_model('chatbot_model.h5')

# Токенизация вопроса пользователя
user_input = "Привет, как дела?"
user_sequence = tokenizer.texts_to_sequences([user_input])
user_padded_sequence = pad_sequences(user_sequence, maxlen=padded_sequences.shape[1])

# Получение ответа от модели
prediction = model.predict(user_padded_sequence)
predicted_index = tf.argmax(prediction, axis=2).numpy()[0][-1]

# Вывод ответа
print("Ответ чатбота:", tokenizer.index_word[predicted_index])

