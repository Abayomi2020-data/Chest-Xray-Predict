from flask import Flask, render_template, request, url_for
from keras.models import load_model 
from keras.preprocessing import image
import numpy as np 
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

dic = {0: 'NORMAL', 1 : 'PNEUMONIA'}

model = load_model("model.h5")

model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(100,100))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1,100,100,3)
    predict_x=model.predict(i) 
    classes_x=np.argmax(predict_x,axis=1)


    return dic[classes_x[0]]

#route 
import os
from flask import send_from_directory

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/jpeg')


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "This is a web app develop by Abayomi Abiodun,  A Data Scientist. follow me on Linkedin: https://www.linkedin.com/in/abayomi-abiodun-49201a1ab/"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]

        img_path = ("static/" + img.filename)
        img.save(img_path)
        
        
        classes_x = predict_label(img_path)
    return render_template("index.html", prediction = classes_x, img_path = img_path)


if __name__ == "__main__":
    app.run(debug=True)