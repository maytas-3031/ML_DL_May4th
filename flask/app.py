from flask import Flask , render_template ,request
import numpy as np
import tensorflow as tf
from keras.models import load_model
global model,graph
graph = tf.get_default_graph()
app  = Flask(__name__)
model = load_model("electric.h5")
@app.route('/')
def hello_world():
    return render_template('base.html')
@app.route('/login', methods = ['GET','POST'])
def login():
    at = request.form["at"]
    v = request.form["v"]
    ap = request.form["ap"]
    rh = request.form["rh"]
    total= [[int(at),int(v),int(ap),int(rh)]]
    print(total)
    with graph.as_default():
        y_pred=model.predict(np.array(total))
    return render_template('base.html',label = "The Electric Power Output is = "+str(y_pred[0][0]))
if __name__=='__main__':
    app.run(debug = True)