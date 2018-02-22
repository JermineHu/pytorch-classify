import base64
import os
from flask import Flask
from train import train
import json
from prediction.classify import classify
from flask_cors import CORS

app = Flask(__name__,static_folder="assets")
CORS(app, supports_credentials=True) # to set cross domain

@app.route('/train/<json_str>',methods=['Get'])
def train(json_str):
    json_data=json.loads(json_str,encoding='utf-8')
    epoch=json_data.get('epoch',30)
    bs=json_data.get('batch_size',32)
    num_workers=json_data.get('num_workers',8)
    lr=json_data.get('learn_rate',0.001)
    fileName=json_data.get('fileName','resNet')
    model_path = train.train(Epoch=epoch, Bs=bs, Num_workers=num_workers, Lr=lr, FileName=fileName)

@app.route("/classify/<image_path>",methods=['Get'])
def classification(image_path):

    image_path= base64.urlsafe_b64decode(image_path).decode('utf-8')
    model_path=os.environ['MODEL_PATH']
    print(image_path,model_path)
    result = classify(img_dir_path=image_path, model_path=model_path, batch_size=32)
    return json.dumps(result)

if __name__ == '__main__':
    app.run(port=int(os.environ['API_PORT']),host='0.0.0.0',debug=True)
