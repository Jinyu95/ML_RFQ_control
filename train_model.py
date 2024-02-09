from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from LSTM_Koopman_model import encoder, Koopman, decoder

def LoadModelandScalers(prefix):
    dir = prefix + '/model.h5'
    model = tf.keras.models.load_model(dir, custom_objects={'encoder': encoder, 'Koopman': Koopman, 'decoder': decoder})
    sc_x = joblib.load(prefix+'/sc_x.pkl')
    sc_d = joblib.load(prefix+'/sc_d.pkl')
    sc_i = joblib.load(prefix+'/sc_i.pkl')

    return model, sc_x, sc_d, sc_i

if __name__ == '__main__':
    input_length = 30  # length of the input sequence
    prediction_length = 30 # length of the prediction sequence
    ndelay = 10 # number of delayed variables
    ninstant = 3 # number of non-delayed variables
    nk = 64 # number of Koopman modes
    
    epochs = 2000
    batch_size = 512
    
    # the data is preprocessed and saved in a .npz file
    loaded_data = np.load("data_arrays.npz")
    delay_data = loaded_data['delay_data']
    instant_data = loaded_data['instant_data']
    frq_data = loaded_data['frq_data']
    future_frq = loaded_data['future_frq'] 
    ntrain = len(delay_data)
    train_number = int(np.round(ntrain * 0.8))
          
    # load the scalers
    sc_x = joblib.load('sc_x.pkl')
    sc_d = joblib.load('sc_d.pkl')
    sc_i = joblib.load('sc_i.pkl')
    
    # build the model
    nfreq = prediction_length
    E = encoder(nk) # encoder
    D = decoder(prediction_length) # decoder
    K = Koopman(nk, prediction_length) # Koopman operator

    inputs_delay = tf.keras.Input(shape=(input_length + prediction_length, ndelay))
    inputs_instant = tf.keras.Input(ninstant)
    input_freq = tf.keras.Input(nfreq)
    encoded, combined_encoded = E(inputs_delay, inputs_instant, input_freq)
    advanced_encoded = K(combined_encoded)
    advanced_freq = D(advanced_encoded)

    model = tf.keras.Model(inputs=[inputs_delay, inputs_instant, input_freq], outputs=advanced_freq)


    lr = 0.001
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')
    
    his = model.fit([delay_data[:train_number], instant_data[:train_number], frq_data[:train_number]],
                    future_frq[:train_number], epochs=epochs, batch_size=batch_size, 
                    validation_data=([delay_data[train_number:], instant_data[train_number:], frq_data[train_number:]],
                                 future_frq[train_number:]), verbose=0)
    model.save('model.h5')