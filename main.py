
from distutils.log import debug
from fileinput import filename
from flask import *  
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import numpy as np
from io import BytesIO 
from keras.optimizers import SGD


app = Flask(__name__)  
  
model = load_model('best_model.h5')
train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip=True, preprocessing_function=preprocess_input)

train = train_datagen.flow_from_directory(directory="train_data",
                                          target_size=(256,256),
                                          batch_size=32)

ref = dict(zip(list(train.class_indices.values()),list(train.class_indices.keys())))

@app.route('/')  
def main():  
    return render_template("upload_image.html")  
  
@app.route('/success', methods = ['POST'])  
def success(): 

    if request.method == 'POST':  
        img = request.files['image']
        img_bytes = BytesIO(img.read())
        img = load_img(img_bytes, target_size=(256,256))
        i = img_to_array(img)
        im = preprocess_input(i)
        img = np.expand_dims(im, axis=0)
        predicton_result = np.argmax(model.predict(img))
        # f.save(f.filename)  
        return render_template("upload_image.html", result = {ref[predicton_result]})  
  
if __name__ == '__main__':  
    app.run(debug=True)

