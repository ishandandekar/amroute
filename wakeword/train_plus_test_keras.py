# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Introduction

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Emergency Vehicle 🚑🚒🚗 Siren 🚨 Sound 🔊 Classification 📢
#
# ### Context
#
# This dataset was completely made manually by extracting audio from sources available on internet sites like Google and Youtube and saved as .wav audio format.
#
# ### Content
#
# The dataset consists of wav-format audio files which are of length 3-seconds. They contain the siren sound of Emergency Vehicles - Ambulance and Firetruck. A third category named Traffic also exists where it contains 3-second .wav format audio files of plain traffic sound. Each category contains 200 sound files, 200 Spectrogram images for each audio file and a python script for converting each audio file to its spectrogram.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Importing Libraries

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:28:31.517875Z","iopub.execute_input":"2026-03-24T17:28:31.519146Z","iopub.status.idle":"2026-03-24T17:28:40.096386Z","shell.execute_reply.started":"2026-03-24T17:28:31.519045Z","shell.execute_reply":"2026-03-24T17:28:40.094970Z"},"jupyter":{"outputs_hidden":false}}
# Audio Processing Libraries
import librosa
import librosa.display
from scipy import signal

# For Playing Audios

# Array Processing
import numpy as np

# Data Visualization

# Display the confusion matrix
from sklearn.metrics import confusion_matrix

# Deal with .pkl files
import pickle

# Create a dataframe
import pandas as pd

# Transform and encode the categorical targets
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Split dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Exploring Dataset

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 🚑 Ambulance

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:28:40.099096Z","iopub.execute_input":"2026-03-24T17:28:40.100329Z","iopub.status.idle":"2026-03-24T17:28:42.533312Z","shell.execute_reply.started":"2026-03-24T17:28:40.100281Z","shell.execute_reply":"2026-03-24T17:28:42.532198Z"},"jupyter":{"outputs_hidden":false}}
filename = "../input/emergency-vehicle-siren-sounds/sounds/ambulance/sound_1.wav"
data, sample_rate = librosa.load(filename)
librosa.display.waveshow(data, sr=sample_rate)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 🚒 Firetruck

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:28:42.535274Z","iopub.execute_input":"2026-03-24T17:28:42.535830Z","iopub.status.idle":"2026-03-24T17:28:43.362675Z","shell.execute_reply.started":"2026-03-24T17:28:42.535784Z","shell.execute_reply":"2026-03-24T17:28:43.361409Z"},"jupyter":{"outputs_hidden":false}}
filename = "../input/emergency-vehicle-siren-sounds/sounds/firetruck/sound_201.wav"
data, sample_rate = librosa.load(filename)
librosa.display.waveshow(data, sr=sample_rate)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 🚕 🚦 Traffic

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:28:43.364710Z","iopub.execute_input":"2026-03-24T17:28:43.366263Z","iopub.status.idle":"2026-03-24T17:28:44.090176Z","shell.execute_reply.started":"2026-03-24T17:28:43.366207Z","shell.execute_reply":"2026-03-24T17:28:44.088614Z"},"jupyter":{"outputs_hidden":false}}
filename = "../input/emergency-vehicle-siren-sounds/sounds/traffic/sound_401.wav"
data, sample_rate = librosa.load(filename)
librosa.display.waveshow(data, sr=sample_rate)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Data Preprocessing

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Using the function features_extractor to get a 80 MFCCs from each audio


# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:28:44.094073Z","iopub.execute_input":"2026-03-24T17:28:44.094535Z","iopub.status.idle":"2026-03-24T17:28:44.101082Z","shell.execute_reply.started":"2026-03-24T17:28:44.094500Z","shell.execute_reply":"2026-03-24T17:28:44.099696Z"},"jupyter":{"outputs_hidden":false}}
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Now we iterate through every audio file and extract features using Mel-Frequency Cepstral Coefficients

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:29:13.689994Z","iopub.execute_input":"2026-03-24T17:29:13.691131Z","iopub.status.idle":"2026-03-24T17:30:02.333288Z","shell.execute_reply.started":"2026-03-24T17:29:13.691089Z","shell.execute_reply":"2026-03-24T17:30:02.329827Z"},"jupyter":{"outputs_hidden":false}}
audio_dataset_path = "../input/emergency-vehicle-siren-sounds/sounds/"

extracted_features = []
for path in os.listdir(audio_dataset_path):
    # Only need ambulance and traffic
    if path == "firetruck":
        continue
    for file in tqdm(
        os.listdir(audio_dataset_path + path + "/"), desc="Extracting " + str(path)
    ):
        if file.lower().endswith(".wav"):
            file_name = audio_dataset_path + path + "/" + file
            data = features_extractor(file_name)
            extracted_features.append([data, path])

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Save the data frame into a .pkl file

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.339267Z","iopub.execute_input":"2026-03-24T17:30:02.340357Z","iopub.status.idle":"2026-03-24T17:30:02.364422Z","shell.execute_reply.started":"2026-03-24T17:30:02.340293Z","shell.execute_reply":"2026-03-24T17:30:02.362991Z"},"jupyter":{"outputs_hidden":false}}
f = open("./Extracted_Features.pkl", "wb")
pickle.dump(extracted_features, f)
f.close()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Read the Extracted_Features from the .pkl file

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.369364Z","iopub.execute_input":"2026-03-24T17:30:02.373811Z","iopub.status.idle":"2026-03-24T17:30:02.388280Z","shell.execute_reply.started":"2026-03-24T17:30:02.371715Z","shell.execute_reply":"2026-03-24T17:30:02.386592Z"},"jupyter":{"outputs_hidden":false}}
f = open("./Extracted_Features.pkl", "rb")
Data = pickle.load(f)
f.close()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Transform Data into a dataframe

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.393484Z","iopub.execute_input":"2026-03-24T17:30:02.395219Z","iopub.status.idle":"2026-03-24T17:30:02.443211Z","shell.execute_reply.started":"2026-03-24T17:30:02.395105Z","shell.execute_reply":"2026-03-24T17:30:02.441812Z"},"jupyter":{"outputs_hidden":false}}
df = pd.DataFrame(Data, columns=["feature", "class"])
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.445774Z","iopub.execute_input":"2026-03-24T17:30:02.446104Z","iopub.status.idle":"2026-03-24T17:30:02.464686Z","shell.execute_reply.started":"2026-03-24T17:30:02.446075Z","shell.execute_reply":"2026-03-24T17:30:02.463168Z"},"jupyter":{"outputs_hidden":false}}
df["class"].value_counts()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Splitting the data into train and test sets

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.466727Z","iopub.execute_input":"2026-03-24T17:30:02.467270Z","iopub.status.idle":"2026-03-24T17:30:02.478556Z","shell.execute_reply.started":"2026-03-24T17:30:02.467234Z","shell.execute_reply":"2026-03-24T17:30:02.477145Z"},"jupyter":{"outputs_hidden":false}}
X = np.array(df["feature"].tolist())
Y = np.array(df["class"].tolist())

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.481169Z","iopub.execute_input":"2026-03-24T17:30:02.481582Z","iopub.status.idle":"2026-03-24T17:30:02.494817Z","shell.execute_reply.started":"2026-03-24T17:30:02.481513Z","shell.execute_reply":"2026-03-24T17:30:02.493448Z"},"jupyter":{"outputs_hidden":false}}
X.shape

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.496654Z","iopub.execute_input":"2026-03-24T17:30:02.497094Z","iopub.status.idle":"2026-03-24T17:30:02.513860Z","shell.execute_reply.started":"2026-03-24T17:30:02.497061Z","shell.execute_reply":"2026-03-24T17:30:02.512344Z"},"jupyter":{"outputs_hidden":false}}
Y.shape

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Label Encoding

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.515630Z","iopub.execute_input":"2026-03-24T17:30:02.516603Z","iopub.status.idle":"2026-03-24T17:30:02.532718Z","shell.execute_reply.started":"2026-03-24T17:30:02.516550Z","shell.execute_reply":"2026-03-24T17:30:02.531244Z"},"jupyter":{"outputs_hidden":false}}
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(Y))
y = np.array(df["class"].map({"ambulance": 1, "traffic": 0}).tolist())
y = to_categorical(y)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.536570Z","iopub.execute_input":"2026-03-24T17:30:02.536978Z","iopub.status.idle":"2026-03-24T17:30:02.548203Z","shell.execute_reply.started":"2026-03-24T17:30:02.536942Z","shell.execute_reply":"2026-03-24T17:30:02.547045Z"}}
labelencoder.classes_
labelencoder.get_params()

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.549728Z","iopub.execute_input":"2026-03-24T17:30:02.550195Z","iopub.status.idle":"2026-03-24T17:30:02.563418Z","shell.execute_reply.started":"2026-03-24T17:30:02.550152Z","shell.execute_reply":"2026-03-24T17:30:02.562174Z"}}
dir(labelencoder)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.565039Z","iopub.execute_input":"2026-03-24T17:30:02.565392Z","iopub.status.idle":"2026-03-24T17:30:02.584219Z","shell.execute_reply.started":"2026-03-24T17:30:02.565361Z","shell.execute_reply":"2026-03-24T17:30:02.582844Z"},"jupyter":{"outputs_hidden":false}}
Y[0]

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.585653Z","iopub.execute_input":"2026-03-24T17:30:02.586527Z","iopub.status.idle":"2026-03-24T17:30:02.599785Z","shell.execute_reply.started":"2026-03-24T17:30:02.586487Z","shell.execute_reply":"2026-03-24T17:30:02.598354Z"}}
set(Y)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.601450Z","iopub.execute_input":"2026-03-24T17:30:02.601931Z","iopub.status.idle":"2026-03-24T17:30:02.614160Z","shell.execute_reply.started":"2026-03-24T17:30:02.601886Z","shell.execute_reply":"2026-03-24T17:30:02.612766Z"},"jupyter":{"outputs_hidden":false}}
y[0]

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.615777Z","iopub.execute_input":"2026-03-24T17:30:02.616240Z","iopub.status.idle":"2026-03-24T17:30:02.630728Z","shell.execute_reply.started":"2026-03-24T17:30:02.616198Z","shell.execute_reply":"2026-03-24T17:30:02.629095Z"}}
y[0:10]

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.632564Z","iopub.execute_input":"2026-03-24T17:30:02.633111Z","iopub.status.idle":"2026-03-24T17:30:02.649034Z","shell.execute_reply.started":"2026-03-24T17:30:02.633064Z","shell.execute_reply":"2026-03-24T17:30:02.647523Z"},"jupyter":{"outputs_hidden":false}}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True
)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.650627Z","iopub.execute_input":"2026-03-24T17:30:02.651070Z","iopub.status.idle":"2026-03-24T17:30:02.659893Z","shell.execute_reply.started":"2026-03-24T17:30:02.651022Z","shell.execute_reply":"2026-03-24T17:30:02.658541Z"},"jupyter":{"outputs_hidden":false}}
y_train.shape

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Display the shape of each splits

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.661903Z","iopub.execute_input":"2026-03-24T17:30:02.662450Z","iopub.status.idle":"2026-03-24T17:30:02.671230Z","shell.execute_reply.started":"2026-03-24T17:30:02.662414Z","shell.execute_reply":"2026-03-24T17:30:02.669851Z"},"jupyter":{"outputs_hidden":false}}
X_train.shape

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.672683Z","iopub.execute_input":"2026-03-24T17:30:02.673313Z","iopub.status.idle":"2026-03-24T17:30:02.684532Z","shell.execute_reply.started":"2026-03-24T17:30:02.673278Z","shell.execute_reply":"2026-03-24T17:30:02.683009Z"},"jupyter":{"outputs_hidden":false}}
X_test.shape

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.686164Z","iopub.execute_input":"2026-03-24T17:30:02.686662Z","iopub.status.idle":"2026-03-24T17:30:02.700800Z","shell.execute_reply.started":"2026-03-24T17:30:02.686618Z","shell.execute_reply":"2026-03-24T17:30:02.699258Z"},"jupyter":{"outputs_hidden":false}}
y_test.shape

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Model Building

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.702288Z","iopub.execute_input":"2026-03-24T17:30:02.703181Z","iopub.status.idle":"2026-03-24T17:30:02.717199Z","shell.execute_reply.started":"2026-03-24T17:30:02.703146Z","shell.execute_reply":"2026-03-24T17:30:02.715815Z"},"jupyter":{"outputs_hidden":false}}
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras import backend as K
from sklearn import metrics
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 LSTM

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.718958Z","iopub.execute_input":"2026-03-24T17:30:02.719425Z","iopub.status.idle":"2026-03-24T17:30:02.738091Z","shell.execute_reply.started":"2026-03-24T17:30:02.719378Z","shell.execute_reply":"2026-03-24T17:30:02.736486Z"},"jupyter":{"outputs_hidden":false}}
x_train_features = X_train.reshape(len(X_train), -1, 80)
x_test_features = X_test.reshape(len(X_test), -1, 80)
print("Reshaped Array Size", x_train_features.shape)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.740624Z","iopub.execute_input":"2026-03-24T17:30:02.741141Z","iopub.status.idle":"2026-03-24T17:30:02.754172Z","shell.execute_reply.started":"2026-03-24T17:30:02.741096Z","shell.execute_reply":"2026-03-24T17:30:02.752875Z"},"jupyter":{"outputs_hidden":false}}
# def lstm(x_tr):
#     K.clear_session()
#     inputs = Input(shape=(x_tr.shape[1], x_tr.shape[2]))
#     #lstm
#     x = LSTM(128)(inputs)
#     x = Dropout(0.5)(x)
#     #dense
#     x = Dense(64, activation='relu')(x)
#     x = Dense(y_test.shape[1], activation='softmax')(x)
#     model = Model(inputs, x)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
#     return model


# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.755898Z","iopub.execute_input":"2026-03-24T17:30:02.756366Z","iopub.status.idle":"2026-03-24T17:30:02.768601Z","shell.execute_reply.started":"2026-03-24T17:30:02.756298Z","shell.execute_reply":"2026-03-24T17:30:02.767198Z"}}
def lstm(x_tr):
    K.clear_session()
    model = Sequential()
    model.add(LSTM(128, input_shape=(x_tr.shape[1], x_tr.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(y_test.shape[1], activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    return model


# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:02.770543Z","iopub.execute_input":"2026-03-24T17:30:02.771037Z","iopub.status.idle":"2026-03-24T17:30:03.192207Z","shell.execute_reply.started":"2026-03-24T17:30:02.771003Z","shell.execute_reply":"2026-03-24T17:30:03.190842Z"},"jupyter":{"outputs_hidden":false}}
model_lstm = lstm(x_train_features)
model_lstm.summary()

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:03.194028Z","iopub.execute_input":"2026-03-24T17:30:03.194477Z","iopub.status.idle":"2026-03-24T17:30:04.334668Z","shell.execute_reply.started":"2026-03-24T17:30:03.194434Z","shell.execute_reply":"2026-03-24T17:30:04.332585Z"},"jupyter":{"outputs_hidden":false}}
plot_model(model_lstm, show_shapes=True, show_layer_names=True)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:04.336948Z","iopub.execute_input":"2026-03-24T17:30:04.337578Z","iopub.status.idle":"2026-03-24T17:30:04.344216Z","shell.execute_reply.started":"2026-03-24T17:30:04.337536Z","shell.execute_reply":"2026-03-24T17:30:04.342891Z"},"jupyter":{"outputs_hidden":false}}
mc = ModelCheckpoint(
    "best_lstm.model.keras",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="max",
)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:04.350248Z","iopub.execute_input":"2026-03-24T17:30:04.351251Z","iopub.status.idle":"2026-03-24T17:30:13.392861Z","shell.execute_reply.started":"2026-03-24T17:30:04.351213Z","shell.execute_reply":"2026-03-24T17:30:13.391611Z"},"jupyter":{"outputs_hidden":false}}
history = model_lstm.fit(
    x_train_features,
    y_train,
    epochs=100,
    callbacks=[mc],
    batch_size=64,
    validation_data=(x_test_features, y_test),
)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:13.394498Z","iopub.execute_input":"2026-03-24T17:30:13.395210Z","iopub.status.idle":"2026-03-24T17:30:13.613838Z","shell.execute_reply.started":"2026-03-24T17:30:13.395176Z","shell.execute_reply":"2026-03-24T17:30:13.612348Z"},"jupyter":{"outputs_hidden":false}}
# summarize history for loss

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:13.615368Z","iopub.execute_input":"2026-03-24T17:30:13.615919Z","iopub.status.idle":"2026-03-24T17:30:13.693597Z","shell.execute_reply.started":"2026-03-24T17:30:13.615886Z","shell.execute_reply":"2026-03-24T17:30:13.692531Z"},"jupyter":{"outputs_hidden":false}}
_, acc = model_lstm.evaluate(x_test_features, y_test)
print("Accuracy:", acc)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:13.695007Z","iopub.execute_input":"2026-03-24T17:30:13.695353Z","iopub.status.idle":"2026-03-24T17:30:14.313595Z","shell.execute_reply.started":"2026-03-24T17:30:13.695321Z","shell.execute_reply":"2026-03-24T17:30:14.312220Z"},"jupyter":{"outputs_hidden":false}}
y_pred = model_lstm.predict(x_test_features)

conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:14.315456Z","iopub.execute_input":"2026-03-24T17:30:14.315983Z","iopub.status.idle":"2026-03-24T17:30:14.349622Z","shell.execute_reply.started":"2026-03-24T17:30:14.315883Z","shell.execute_reply":"2026-03-24T17:30:14.348205Z"},"jupyter":{"outputs_hidden":false}}
model_lstm.save("./sireNN.keras")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-24T17:30:14.351349Z","iopub.execute_input":"2026-03-24T17:30:14.351820Z","iopub.status.idle":"2026-03-24T17:30:15.209528Z","shell.execute_reply.started":"2026-03-24T17:30:14.351775Z","shell.execute_reply":"2026-03-24T17:30:15.208144Z"}}
import keras

reconstructed_model = keras.models.load_model("./sireNN.keras")
_, acc = reconstructed_model.evaluate(x_test_features, y_test)
print("Accuracy:", acc)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # 🚨 Conclusion
#
# **In this notebook we learnt how to work with Sound Data and perform Deep Learning Methods for classification by training a CNN as well as an LSTM.**
#
# **Please feel free to follow me on Kaggle and upvote this Notebook if you found it useful : )**
#
# **📕📕📕 Happy Learning !!! 📕📕📕**
