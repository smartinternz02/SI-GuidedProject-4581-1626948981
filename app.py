from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
import pickle 
import tensorflow as tf
graph = tf.compat.v1.get_default_graph()
with open(r'count_vec.pkl','rb') as file:
    cv=pickle.load(file)

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/',methods=["POST"])
def prediction():
        topic = request.form['review']
        print("Hey " + topic)
        topic1=cv.transform([topic])
        
        print("\n"+str(topic1.shape)+"\n")
        with graph.as_default():
             model = load_model('phone.h5')
             model.compile(optimizer='adam',loss='binary_crossentropy')
             y_pred = model.predict(topic1)
             print("pred is "+str(y_pred))
        if(y_pred > 0.5):
            topic2 = "Positive Review"
        else:
            topic2 = "Negative Review"

        return render_template('index.html',ypred = topic2)
        


if __name__=='__main__':
    app.run(debug=True)

