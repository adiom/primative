import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.cluster import KMeans

# Загрузка данных из файла
with open('c_training_data.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Разделение на вопросы и ответы
questions = [line.strip() for i, line in enumerate(lines) if i % 2 == 0]
answers = [line.strip() for i, line in enumerate(lines) if i % 2 != 0]

# Инициализация токенизатора
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

# Преобразование текстов в последовательности чисел
questions_sequences = tokenizer.texts_to_sequences(questions)
answers_sequences = tokenizer.texts_to_sequences(answers)

# Определение максимальной длины последовательности
max_len = max(len(seq) for seq in questions_sequences + answers_sequences)

# Приведение всех последовательностей к одной длине
questions_padded_sequences = pad_sequences(questions_sequences, maxlen=max_len, padding='post')
answers_padded_sequences = pad_sequences(answers_sequences, maxlen=max_len, padding='post')

# Кластеризация вопросов
num_clusters = 5  # Выберите количество кластеров
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
question_clusters = kmeans.fit_predict(questions_padded_sequences)

# Создание модели для каждого кластера
models = []
for cluster_id in range(num_clusters):
    # Получение индексов вопросов в текущем кластере
    cluster_indices = np.where(question_clusters == cluster_id)[0]
    
    # Выделение данных для текущего кластера
    cluster_questions = questions_padded_sequences[cluster_indices]
    cluster_answers = answers_padded_sequences[cluster_indices]
    
    # Создание модели
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_len),
        LSTM(16),
        Dense(len(tokenizer.word_index) + 1, activation='softmax')
    ])
    
    # Компиляция модели
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Обучение модели
    model.fit(cluster_questions, np.argmax(cluster_answers, axis=1), epochs=50)
    
    # Сохранение модели
    model.save(f'cluster_{cluster_id}_model.h5')
    models.append(model)

