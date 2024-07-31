import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hank', 'Sten']
genders = ['female', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'male']

gender_mapping = {'male': 0, 'female': 1}
y = np.array([gender_mapping[gender] for gender in genders])

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(names)
X = tokenizer.texts_to_sequences(names)
print(X)
X = pad_sequences(X, padding='post')
print(X)
print(tokenizer.word_index)
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=1)

def predict_gender(name):
    seq = tokenizer.texts_to_sequences([name])
    padded_seq = pad_sequences(seq, maxlen=X.shape[1], padding='post')
    prediction = model.predict(padded_seq)
    return prediction[0][0]

test_name = 'Lina'
predicted_gender = predict_gender(test_name)
print(f"Das vorhergesagte Geschlecht für den Namen {test_name} ist: {int((1-predicted_gender)*100)}% männlich und zu {int(predicted_gender*100)}% weiblich")
