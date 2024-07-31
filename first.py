import keras
import visualkeras
import numpy as np
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import TensorBoard

xl = []
yl = []

for i in range(0, 10000):
    xl.append(i)
    yl.append(i*2)


X = np.array(xl)
y = np.array(yl)

model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(10, input_dim=1))  # Eingabeschicht mit 10 Neuronen
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))  # Ausgabeschicht

log_dir = "logs/fit/"  # Verzeichnis für TensorBoard-Logs
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)


model.compile(optimizer='adam', loss='mean_squared_error')

#Epoche: Vollständiger Durchlauf über Dataset
#Batch: Auswahl aus dem Dataset mit Schritten innerhalb der Epoche

model.fit(X, y, epochs=200, verbose=1)

#epochs = 100
#steps_per_epoch = 5  # Anzahl der Fittings pro Epoche

#for epoch in range(epochs):
#    for step in range(steps_per_epoch):
#        loss = model.train_on_batch(X, y)
#    if (epoch + 1) % 100 == 0:  # Ausgabe des Verlusts alle 100 Epochen
#        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')
print(model.summary())
model.save('model.h5')
model.save_weights('model.weights.h5')

visualkeras.layered_view(model, to_file='model_plot.png').show()
print("Die nächste Zahl in der Reihenfolge ist:", model.predict(np.array([7]), verbose=0))
print("Die nächste Zahl in der Reihenfolge ist:", model.predict(np.array([22]), verbose=0))
print("Die nächste Zahl in der Reihenfolge ist:", model.predict(np.array([33]), verbose=0))