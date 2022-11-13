import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Reshape,
)
from tcn import TCN

# OPTUNA MODELS
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.samplers import TPESampler

import os

DATA_DIR = Path(__file__).parent / ".." / "data" / "hpo"

tweet_data = pd.read_csv(DATA_DIR / "agg_btc_vader.csv", index_col=0)
stock_data = pd.DataFrame(pd.read_csv(DATA_DIR / "labels_v2_s96.csv", index_col=0)["btc"])
tweet_counts = pd.read_csv(DATA_DIR / "btc_tweet_counts.csv", index_col=0)
scaler_df = pd.read_csv(DATA_DIR / "scale_s96.csv", index_col=0)

merged = stock_data.merge(tweet_data, left_index=True, right_index=True, how="left")
counts_added = merged.merge(tweet_counts, left_index=True, right_index=True, how="left")
merged = counts_added
merged = merged.fillna(0)

def squared_epsilon_insensitive_loss(epsilon=0.025):
    def _loss(y_true, y_pred):
        losses = tf.where(
            tf.abs(y_true - y_pred) > epsilon,
            tf.square(tf.abs(y_true - y_pred) - epsilon),
            0,
        )
        return tf.reduce_sum(losses)

    return _loss


def calculate_RV(signal):
    """Calcualte RV from log returns
    Parameters
    ----------
    signal : np.ndarray
        Input array with first dimension as days and last dimension as log-returns
    Returns
    -------
    np.ndarray
        [description]
    """
    return 100 * np.sqrt(np.square(signal).sum(axis=-1))


def evaluate_forecast(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))
    me = np.mean(forecast - actual)
    mae = np.mean(np.abs(forecast - actual))
    mpe = np.mean((forecast - actual) / actual)
    rmse = np.mean((forecast - actual) ** 2) ** 0.5
    msle = np.mean(np.square(np.log(1 + forecast) - np.log(1 + actual)))
    return {"mape": mape, "me": me, "mae": mae, "mpe": mpe, "rmse": rmse, "msle": msle}

def inv_transform(sig, df_scale=scaler_df):
    """Inverse transform scaled signals
    Parameters
    ----------
    sig : np.ndarray; (_days, _steps, _channels)
        Signals wrt given dimensions

    df_scale : DataFrame
        Respective scale dataframe with columns mins and scales necessary
        for inverse_transform function.

    Returns
    -------
    to_ret : np.ndarray; (_days, _steps, _channels)
        Converted back to Log-Return Scales
    """
    to_ret = np.zeros((sig.shape[0], sig.shape[1], sig.shape[2]))
    print(np.shape(to_ret))
    for i in range(sig.shape[0]):
        for j in range(sig.shape[2]):
            scaler = MinMaxScaler()
            scaler.min_, scaler.scale_ = df_scale["mins"][j], df_scale["scales"][j]
            to_ret[i,:,j] = scaler.inverse_transform(sig[i,:,j].reshape(sig.shape[1],1)).reshape(sig.shape[1],)

    return to_ret

def inv_transform_2(sig,df_scale=scaler_df):
    scalermin_, scalerscale_ = df_scale["mins"][0], df_scale["scales"][0]
    X = tf.reshape(sig[0,:,0],[sig.shape[1],1])
    X -= scalermin_
    X /= scalerscale_
    return tf.reshape(X, [1,96,1])

def calculate_RV_2(signal):
    return 100 * tf.math.sqrt(tf.math.reduce_sum(tf.math.square(signal),axis=-1))

def mape(y_true, y_pred):
    rv_true = calculate_RV_2(inv_transform_2(y_true)[:,:,0])
    rv_pred = calculate_RV_2(inv_transform_2(y_pred)[:,:,0])
    mape = tf.reduce_mean(tf.abs((rv_true - rv_pred) / rv_true))
    return mape
mape.__name__ = "mape"

train_x = merged["btc"].values[:6912].reshape(72,96,1)
val_x = merged["btc"].values[6912:9216].reshape(24,96,1)
full_train_x = merged["btc"].values[:9216].reshape(96,96,1)
test_x = merged["btc"].values[9216:13824].reshape(48,96,1)

train_y = merged["btc"].values[96:7008].reshape(72,96,1)
val_y = merged["btc"].values[7008:9312].reshape(24,96,1)
full_train_y = merged["btc"].values[96:9312].reshape(96,96,1)
test_y = merged["btc"].values[9312:13920].reshape(48,96,1)


if len(sys.argv) > 1:
    try:
        mode = int(sys.argv[1])
        if mode not in [0,1,2]:
            raise ValueError
    except:
        raise ValueError(f"mode must be one of 0,1,2 not {sys.argv[1]}")
else:
    raise ValueError(f"mode must be specified")

MODE_NAMES = {
    0: "TCN",
    1: "GRU",
    2: "LSTM"
}
MODE_GPUS = {
    0: "3",
    1: "4",
    2: "5"
}

mode_name = MODE_NAMES[mode]
os.environ["CUDA_VISIBLE_DEVICES"] = MODE_GPUS[mode]

#mode 0 = TCN, mode 1 = GRU, mode 2 = LSTM
def create_model(trial, mode=mode):

    # Hyperparameters to be tuned by Optuna.
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-2, log=True)
    rec_units = trial.suggest_int("recurrent", 32, 512)
    dropout_units = trial.suggest_uniform("dropout", 0, 0.5)
    epsilon = trial.suggest_float("epsilon", 1e-2, 1e-1, log=True)

    model = tf.keras.Sequential()
    if mode == 0:
        kernel_size = trial.suggest_int("kernel_size", 2, 6)
        dilations_int = trial.suggest_int("dilations", 2, 4)
        skip_conn = trial.suggest_categorical("skip_conn", [True, False])
        b_norm = trial.suggest_categorical("batch_norm", ["batch", "weight", "none", "layer"])
        if b_norm == "batch":
            model.add(
                TCN(
                    rec_units, 
                    kernel_size=kernel_size,
                    dilations=(dilations_int**0,dilations_int**1,dilations_int**2,dilations_int**3,dilations_int**4,dilations_int**5),
                    dropout_rate=dropout_units,
                    use_skip_connections=skip_conn,
                    use_batch_norm=True,
                    use_weight_norm=False,
                    use_layer_norm=False
                )
            )
        elif b_norm == "weight":
            model.add(
                TCN(
                    rec_units, 
                    kernel_size=kernel_size,
                    dilations=(dilations_int**0,dilations_int**1,dilations_int**2,dilations_int**3,dilations_int**4,dilations_int**5),
                    dropout_rate=dropout_units,
                    use_skip_connections=skip_conn,
                    use_batch_norm=False,
                    use_weight_norm=True,
                    use_layer_norm=False
                )
            )
        elif b_norm == "layer":
            model.add(
                TCN(
                    rec_units,
                    kernel_size=kernel_size,
                    dilations=(dilations_int**0,dilations_int**1,dilations_int**2,dilations_int**3,dilations_int**4,dilations_int**5),
                    dropout_rate=dropout_units,
                    use_skip_connections=skip_conn,
                    use_batch_norm=False,
                    use_weight_norm=False,
                    use_layer_norm=True
                )
            )
        else:
            model.add(
                TCN(
                    rec_units,
                    kernel_size=kernel_size,
                    dilations=(dilations_int**0,dilations_int**1,dilations_int**2,dilations_int**3,dilations_int**4,dilations_int**5),
                    dropout_rate=dropout_units,
                    use_skip_connections=skip_conn,
                    use_batch_norm=False,
                    use_weight_norm=False,
                    use_layer_norm=False
                )
            )

    elif mode == 1:
        model.add(GRU(rec_units, dropout=dropout_units))
    else:
        model.add(LSTM(rec_units, dropout=dropout_units))

    model.add(tf.keras.layers.Dense(units=96))
    model.add(Reshape((96,1)))

    # Compile model.
    model.compile(
        optimizer=tfa.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        ),
        loss=squared_epsilon_insensitive_loss(epsilon=epsilon),
        metrics=mape
    )

    return model


def objective(trial, mode=mode, train_x=train_x, train_y=train_y, valid_x=val_x, valid_y=val_y):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    # Metrics to be monitored by Optuna.
    monitor = "val_mape"

    # Create tf.keras model instance.
    model = create_model(trial, mode)

    # Create callbacks for early stopping and pruning.
    callbacks = [
        #tf.keras.callbacks.EarlyStopping(patience=3),
        TFKerasPruningCallback(trial, monitor),
    ]

    # Train model.
    history = model.fit(
        x=train_x,
        y=train_y,
        epochs=30,
        validation_data=(valid_x, valid_y),
        callbacks=callbacks,
        shuffle=False,
        batch_size=1
    )

    return history.history[monitor][-1]

def show_result(study):

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    best_test_result = objective(trial, train_x=full_train_x, train_y=full_train_y, valid_x=test_x, valid_y=test_y)
    print(f"Best result in test set: {best_test_result}")

study = optuna.create_study(
        direction="minimize", sampler=TPESampler(multivariate=True), storage=f"sqlite:///{mode_name}.db",
        load_if_exists=True, pruner=optuna.pruners.NopPruner()
)

study.optimize(objective, n_trials=250, catch=(Exception,))

show_result(study)
