import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Layer,
    Dense,
    GRU,
    LSTM,
    Attention,
    Embedding,
    Reshape,
    TimeDistributed,
    BatchNormalization, Concatenate
)
from tcn import TCN

class VADER_dTCN(Model):
  def __init__(
        self,
        rec_dim_1=64,
        rec_dim_2=8,
        dilations_int = 4,
        kern = 5,
        drop_rate_1 = 0,
        drop_rate_2 = 0,
        dense_dim = 4,
        add_layer_norm = False,
        add_batch_norm = False,
        dense_act = "linear",
        name = "VADER_dTCN",
        **kwargs
    ):
        super(VADER_dTCN, self).__init__(name=name, **kwargs)
        self.dense_1 = TimeDistributed(Dense(dense_dim, activation=dense_act))
        self.batch_norm = TimeDistributed(BatchNormalization())
        self.bn = add_batch_norm
        self.tcn_time = TCN(
            rec_dim_1,
            kernel_size=kern,
            nb_stacks=1,
            dilations=(dilations_int**0,dilations_int**1,dilations_int**2,dilations_int**3,dilations_int**4,dilations_int**5),
            dropout_rate=drop_rate_1,
            use_batch_norm=False,
            use_layer_norm=False,
            use_weight_norm=False,
            use_skip_connections=True,
        )
        self.tcn_tweet = TCN(
            rec_dim_2,
            kernel_size=kern,
            nb_stacks=1,
            dilations=(dilations_int**0,dilations_int**1,dilations_int**2,dilations_int**3,dilations_int**4,dilations_int**5),
            dropout_rate=drop_rate_2,
            use_batch_norm=False,
            use_layer_norm=False,
            use_weight_norm=False,
            use_skip_connections=True,
        )

        self.dense_2 = Dense(96)
        self.reshape = Reshape((96,1))

  def call(self, inputs):
    tweet_inf = inputs[:,:,1:]
    time_inf = Reshape((96,1))(inputs[:,:,0])

    time_x = self.tcn_time(time_inf)
    tweet_x = self.dense_1(tweet_inf)

    if self.bn:
      tweet_x = self.batch_norm(tweet_x)

    tweet_x = self.tcn_tweet(tweet_x)
    pre_dense = Concatenate(axis=1)([tweet_x, time_x])
    x = self.dense_2(pre_dense)
    x = self.reshape(x)

    return x

