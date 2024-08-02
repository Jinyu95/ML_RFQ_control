from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import numpy as np
import LSTM_Koopman_model
import torch
import torch.nn as nn
import torch.nn.functional as F


# def LoadModelandScalers_tensorflow(prefix):
#     dir = prefix + '/model.h5'
#     model = tf.keras.models.load_model(dir, custom_objects={'encoder': encoder, 'Koopman': Koopman, 'decoder': decoder})
#     sc_x = joblib.load(prefix+'/sc_x.pkl')
#     sc_d = joblib.load(prefix+'/sc_d.pkl')
#     sc_i = joblib.load(prefix+'/sc_i.pkl')

#     return model, sc_x, sc_d, sc_i
def LoadModelandScalers_pytorch(prefix):
    dir = prefix + '/model.pth'
    model = LSTM_Koopman_model.KoopmanModel(12, 128, 60)
    model.load_state_dict(torch.load(dir, map_location=torch.device('cpu')))
    scaler_d = joblib.load(prefix+'/scaler_d.pkl')
    scaler_f = joblib.load(prefix+'/scaler_f.pkl')
    return model, scaler_d, scaler_f


if __name__ == '__main__':
    input_length = 60