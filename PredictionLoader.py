import numpy as np
# Model loading. saving and image testing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


## Label dictionnary 1 (classes acronyms):
label_to_class = {
    0 : "akiec",
    1 : "bcc",
    2 : "bkl",
    3 : "df",
    4 : "mel",
    5 : "nv",
    6 : "vasc"
}

## Label dictionnary 2 (actual lesions):
label_to_lesion = {
    0 : "Actinic keratoses and intraepithelial carcinoma",
    1 : "Basal cell carcinoma",
    2 : "Benign keratosis-like",
    3 : "Dermatofibroma",
    4 : "Melanoma",
    5 : "Melanocytic nevi",
    6 : "Vascular lesion"
}

## Key from value in a dict. func. :
def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

## Image size tuple (in pixels, same size used in pre-processing)
size = (32,32)

# Load the model
model = load_model('model850(setup2_32_85).h5')

# Input image for testing
img_path = "Data\\HAM10000\\HAM10000_images\\ISIC_0024309.jpg"

## <--------------------------------------------------------------------------------------------------------------------------->

## Function 1 : GetPrediction with probability
# -> Arguments   : Image path
# -> Output      : Image predicted class
def GetPrediction(path) :
    img = image.load_img(path, target_size=(size[0], size[1])) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image data
    img_array /= 255 

    # Make predictions
    predictions = model.predict(img_array)[0]

    # Get the predicted class label
    predicted_class = np.argmax(predictions)
    
    return f"{label_to_lesion[predicted_class]}:{predictions[predicted_class]}"


## Function 2 : GetPredictionProbs
# -> Arguments   : Image path
# -> Output      : Dict. of  class probabilities
def GetPredictionProbs(path) :
    img = image.load_img(path, target_size=(size[0], size[1])) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image data
    img_array /= 255 

    # Make predictions
    predictions = model.predict(img_array)[0].tolist()
    predictions_sorted = sorted(model.predict(img_array)[0].tolist(), reverse=True)
    return [f"{label_to_lesion[predictions.index(predictions_sorted[i])]}:{predictions_sorted[i]}" for i in range(7)]

## <--------------------------------------------------------------------------------------------------------------------------->

