import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(model, img_file,  class_names):

    img = image.load_img(img_file, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prediction = prediction[0]  # convert (1, num_classes) → (num_classes,)
    prob_dict = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class , prob_dict