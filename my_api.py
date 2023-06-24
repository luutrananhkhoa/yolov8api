from flask import Flask, request, Response, Flask, jsonify
from flask_cors import CORS, cross_origin
from flask import request
import os
import cv2
from ultralytics import YOLO
import json
import math
import random
# Declare Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"
app.static_folder = "static"

@app.route('/upload', methods=['POST'] )
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", 
        passes it through YOLOv8 object detection 
        network and returns an array of bounding boxes.
        :return: a JSON array of objects bounding 
        boxes in format 
        [[x1,y1,x2,y2,object_type,probability],..]
    """
    image = request.files["file"]
    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    print("Save = ", path_to_save)
    image.save(path_to_save)

    frame = cv2.imread(path_to_save)
    boxes = detect_objects_on_image(frame)
    print('boxes',boxes)
    if len(boxes) > 0:
        cv2.imwrite(path_to_save, boxes)
        del frame
        #     'message': 'Data response',
        #     'value': path_to_save
        # }
        # response_data = 'Value: {}'.format(data['value'])
        # return Response(response_data, status=200, mimetype='text/plain')
        file_url = request.host_url.rstrip('/') + '/static/' + image.filename
        data = {
            'message': 'Data response',
            'value': file_url
        }
        response_data = jsonify(data)
        return response_data

   
    return 'Upload file to detect'


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
        cv2.rectangle(buf, (x1,y1), (x2,y2), (255,0,255),1)
        conf=math.ceil((box.conf[0]*100))/100
        class_name=result.names[class_id]
        label=f'{class_name}{conf}'
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        print('t_size',t_size)
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(buf, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
        cv2.putText(buf, label, (x1,y1-2),0, 0.5,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
    return buf


@app.route('/uploadvideo', methods=['POST'] )
def detectVideo():
    video = request.files["file"]
    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    print("Save = ", path_to_save)
    video.save(path_to_save)
    cap=cv2.VideoCapture(path_to_save)
    # breakpoint()
    frame_width=int(cap.get(3))
    frame_height = int(cap.get(4))
    random_number = random.randint(11111, 99999)
    random_number_str = str(random_number)
    output =random_number_str+'_detect.avi'
    output_filename = './static/'+output 
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    model=YOLO("weights/best.pt")
    classNames = ['AppleFresh','AppleRotten','BananaRotten','BananaUnripe','DragonFruitFresh','DragonFruitRotten','GuavaFresh','GuavaRotten','MangoFresh','MangoRotten','MangoUnripe','OrangesFresh','OrangesRotten','OrangesUnripe','PapayaFresh','PapayaRotten','PapayaUnripe','PomegranateFresh','PomegranateRotten','StrawberryFresh','StrawberryUnripe','TomatoFresh','TomatoRotten','TomatoUnripe','BananaFresh','StrawberryRotten']
    
    while True:
        ret, img = cap.read()
        # Doing detections using YOLOv8 frame by frame
        #stream = True will use the generator and it is more efficient than normal
        if ret:
            results=model(img,stream=True)
            #Once we have the results we can check for individual bounding boxes and see how well it performs
            # Once we have have the results we will loop through them and we will have the bouning boxes for each of the result
            # we will loop through each of the bouning box
            for r in results:
                boxes=r.boxes
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0]
                    #print(x1, y1, x2, y2)
                    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                    print(x1,y1,x2,y2)
                    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                    #print(box.conf[0])
                    conf=math.ceil((box.conf[0]*100))/100
                    cls=int(box.cls[0])
                    class_name=classNames[cls]
                    label=f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    #print(t_size)
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
            out.write(img)
            # cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF==ord('1'):
                break
        else:
            break
    out.release()
    cv2.destroyAllWindows()
        # file_url = request.host_url.rstrip('/') + '/static/' + out.filename
    # return cap.filename
    file_url = request.host_url.rstrip('/') + '/static/' + output
    data = {
        'message': 'Data response',
        'value': file_url
    }
    response_data = jsonify(data)
    return response_data

# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')