# from tensorflow.keras import layers
# import tensorflow as tf

# class encoder(tf.keras.layers.Layer):
#     def __init__(self, nencoded, **kwargs):
#         super(encoder, self).__init__(**kwargs)
#         self.nencoded = nencoded
#         self.lstm = layers.LSTM(self.nencoded, return_sequences=False, return_state=False)
#         self.dense = layers.Dense(units=nencoded, activation='relu')

#     def call(self, input_lstm, input_dense, input_freq):
#         lstm_out = self.lstm(input_lstm)
#         dense_out = self.dense(input_dense)
#         encoded = lstm_out + dense_out
#         combined_encoded = tf.keras.layers.Concatenate()([input_freq, encoded])
#         return encoded, combined_encoded
    
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({"nencoded": self.nencoded})
#         return config

# class Koopman(tf.keras.layers.Layer):
#     def __init__(self, nencoded, nfrq, **kwargs):
#         super(Koopman, self).__init__(**kwargs)
#         self.nencoded = nencoded
#         self.nfrq = nfrq
#         self.KoopmanOperator = layers.Dense(units=self.nencoded+self.nfrq, activation=None, use_bias=False)
        
#     def call(self, combined_encoded):
#         advanced_encoded = self.KoopmanOperator(combined_encoded)
#         return advanced_encoded
    
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({"nencoded": self.nencoded, "nfrq": self.nfrq})
#         return config

# class decoder(tf.keras.layers.Layer):
#     def __init__(self, nfreq, **kwargs):
#         super(decoder, self).__init__(**kwargs)
#         self.nfreq = nfreq
        
#     def call(self, advanced_encoded):
#         advanced_freq = advanced_encoded[:, :self.nfreq]
#         return advanced_freq
    
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({"nfreq": self.nfreq})
#         return config

import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, nencoded, input_dim_lstm):
        super(Encoder, self).__init__()
        self.nencoded = nencoded
        self.lstm = nn.LSTM(input_dim_lstm, nencoded, batch_first=True)
        # self.dense = nn.Linear(input_dim_dense, nencoded)

    def forward(self, input_lstm, input_freq):
        _, (lstm_out, _) = self.lstm(input_lstm)
        lstm_out = lstm_out[-1]  # Get the last hidden state
        # dense_out = F.relu(self.dense(input_dense))
        encoded = lstm_out #+ dense_out
        combined_encoded = torch.cat((input_freq, encoded), dim=1)
        return encoded, combined_encoded

class Koopman(nn.Module):
    def __init__(self, nencoded, nfrq):
        super(Koopman, self).__init__()
        self.nencoded = nencoded
        self.nfrq = nfrq
        self.KoopmanOperator = nn.Linear(nencoded + nfrq, nencoded + nfrq, bias=False)

    def forward(self, combined_encoded):
        advanced_encoded = self.KoopmanOperator(combined_encoded)
        return advanced_encoded

class Decoder(nn.Module):
    def __init__(self, nfreq):
        super(Decoder, self).__init__()
        self.nfreq = nfreq

    def forward(self, advanced_encoded):
        advanced_freq = advanced_encoded[:, :self.nfreq]
        return advanced_freq

class KoopmanModel(nn.Module):
    def __init__(self, input_dim_lstm, nencoded, nfreq):
        super(KoopmanModel, self).__init__()
        self.encoder = Encoder(nencoded, input_dim_lstm)
        self.koopman = Koopman(nencoded, nfreq)
        self.decoder = Decoder(nfreq)

    def forward(self, inputs_delay, input_freq):
        encoded, combined_encoded = self.encoder(inputs_delay, input_freq)
        advanced_encoded = self.koopman(combined_encoded)
        advanced_freq = self.decoder(advanced_encoded)
        return advanced_freq