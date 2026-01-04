# TensorFlow 2.x
import numpy as np
import tensorflow as tf

# ----- build a small MLP f_theta(x) -> scalar logit -----
def build_model(input_dim, hidden=(64, 64), l2=1e-4, dropout=0.0):
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu", kernel_regularizer=reg)(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    # scalar output: f_theta(x)
    outputs = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=reg)(x)
    return tf.keras.Model(inputs, outputs)

# ----- custom loss that matches: sum_{X1} log(1+exp(-f)) + sum_{X2} log(1+exp(f)) -----
# With labels y=+1 for X1, y=-1 for X2, this is sum softplus(-y * f)
class LogisticLossSum(tf.keras.losses.Loss):
    def __init__(self, name="logistic_sum"):
        super().__init__(reduction=tf.keras.losses.Reduction.SUM, name=name)

    def call(self, y_true, y_pred):
        # y_true in {+1, -1}, y_pred is shape (batch, 1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.squeeze(y_pred, axis=-1)
        # softplus(z) = log(1 + exp(z))
        return tf.nn.softplus(-y_true * y_pred)

# Optional: accuracy for y∈{+1,-1}
@tf.function
def binary_acc_pm1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.squeeze(y_pred, axis=-1)
    return tf.reduce_mean(
        tf.cast(tf.equal(y_pred > 0.0, y_true > 0.0), tf.float32)
    )

def train_logistic_nn( X1, X2, epochs=200, batch_size=128, lr=1e-3,
                      hidden=(64, 64), l2=1e-4, dropout=0.0, seed=42):
    tf.random.set_seed(seed); np.random.seed(seed)

    # Prepare data: X = [X1; X2], y = [+1...; -1...]
    X1 = np.asarray(X1, dtype=np.float32)
    X2 = np.asarray(X2, dtype=np.float32)
    X = np.vstack([X1, X2]).astype(np.float32)
    y = np.concatenate([np.ones(len(X1), dtype=np.float32),
                        -np.ones(len(X2), dtype=np.float32)])

    model = build_model(X.shape[1], hidden=hidden, l2=l2, dropout=dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=LogisticLossSum(),
        metrics=[binary_acc_pm1]
    )

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )

    return model, history


def f_theta(  X1, X2,x):
    model, hist = train_logistic_nn(X1, X2, epochs=50, batch_size=128, lr=1e-3)
    """
    Compute f_theta(x) from a trained Keras model.
    - x can be shape (d,) for a single point or (n, d) for a batch.
    - returns a float for a single x, or a 1D array for a batch.
    """
    x = np.asarray(x, dtype=np.float32)
    single = (x.ndim == 1)
    if single:
        x = x[None, :]
    logits =  model(x, training=False).numpy().squeeze(-1)  # f_theta(x)
    return float(logits[0]) if single else logits
