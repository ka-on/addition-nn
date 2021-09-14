import tensorflow as tf
import numpy as np

x = np.asarray(tf.random.uniform(
    shape=(30, 2), minval=-100, maxval= 100, dtype=tf.dtypes.int32
))

y = np.array([i[0] + i[1] for i in x], dtype=int)


def build():
    model = tf.keras.models.Sequential(
        tf.keras.layers.Dense(units=1, input_shape=[2]),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss=tf.keras.losses.mean_squared_error
    )
    return model

def train(model):
    history = model.fit(x, y, epochs=5000, verbose=False)
    return history

def main():
    model = build()
    print(model.summary())
    history = train(model)
    while True:
        a = int(input("x: "))
        b = int(input("y: "))
        feed = np.array([[a,b]], dtype=int)
        print(model.predict(feed))

main()
