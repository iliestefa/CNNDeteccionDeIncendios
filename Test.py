from tensorflow import keras
import tensorflow as tf
import numpy as np
from Constantes import class_names, batch_size, img_height, img_width

#Se importa el/los modelo/s creado/s--------------------------------------------------------------------------------------------------------
#model = keras.models.load_model("models/modelIliana.h5")
modelTony = keras.models.load_model("models/modelTony.h5")
#modelSesme = keras.models.load_model("models/modelSesme.h5")


#Se descarga la imagen----------------------------------------------------------------------------------------------------------------------
foto_url = "https://i.pinimg.com/originals/5c/cb/7e/5ccb7e31398e7d8a0a255b4c9f042a52.jpg"
foto_path = tf.keras.utils.get_file('nofuego', origin=foto_url)


#Se procesa la imagen----------------------------------------------------------------------------------------------------------------------
img = keras.preprocessing.image.load_img(
    foto_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch


#Se predice un resultado-------------------------------------------------------------------------------------------------------------------

predictions = modelTony.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "Tony :This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

'''
predictions = modelSesme.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "Sesme :This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "Iliana: This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
'''