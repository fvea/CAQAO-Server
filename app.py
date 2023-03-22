from flask import Flask, request, send_file, jsonify
from PIL import Image

from datetime import datetime
import socket
import torch
import io
import os
import secrets

from db import db_init, db
from model import Detection, TempDetection


DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
IMAGE_URL = "http://{}:5000/detections/{}"
PREVIEW_IMAGE_URL = "http://{}:5000/preview/{}"
TEMP_DIR = "temp"

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
FLASK_IP_ADDR = ip_address

app = Flask(__name__)
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)


@app.route("/assess", methods=["POST"])
def assess():
    
    if request.method == "POST":

        # get cacao beans image and bean size from request
        image_file = request.files.get("image")
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        bean_size = int(request.form["beanSize"])

        # reduce size=640 for faster inference
        results = model(image, size=640)
        results.render()
        image_detection = Image.fromarray(results.ims[0])

        # convert the image detection results to bytes
        with io.BytesIO() as output_bytes:
            image_detection.save(output_bytes, format='JPEG')
            image_detection_bytes = output_bytes.getvalue()

        # get cacao class counts and compute bean grade
        class_counts = get_class_detection_counts(results)
        bean_grade = get_bean_grade(class_counts, bean_size)

        # temporarily save the detection result
        temp_detection = TempDetection(
            image=image_detection_bytes,
            mimetype=image_file.mimetype,
            filename=f"{datetime.now().strftime(DATETIME_FORMAT)}.jpg",
            beanGrade=bean_grade,
            **class_counts
        )
        db.session.add(temp_detection)
        db.session.commit()
        
    return get_json_response(temp_detection)
    

@app.route("/save_results", methods=["POST"])
def save_results():

    if request.method == "POST":
        # save the temporary detection to main detection table
        temp_detection = db.session.query(TempDetection).order_by(TempDetection.id.desc()).first()
        detection = Detection(
            image=temp_detection.image,
            mimetype=temp_detection.mimetype,
            filename=temp_detection.filename,
            beanGrade=temp_detection.beanGrade,
            veryDarkBrown = temp_detection.veryDarkBrown,
            brown = temp_detection.brown,
            partlyPurple = temp_detection.partlyPurple,
            totalPurple = temp_detection.totalPurple,
            g1 = temp_detection.g1,
            g2 = temp_detection.g2,
            g3 = temp_detection.g3,
            g4 = temp_detection.g4,
            mouldy = temp_detection.mouldy,
            insectInfested = temp_detection.insectInfested,
            slaty = temp_detection.slaty,
            germinated = temp_detection.germinated
        )
        # add the new record to the database session and commit the changes
        db.session.add(detection)
        db.session.commit()
        # delete the temporary detections
        db.session.query(TempDetection).delete()
        db.session.commit()

        return "Assessment Results Saved", 200
    
@app.route('/detections')
def get_detections():
    detections = Detection.query.all()
    users_list = [{'id': detection.id, 'filename': f" http://{FLASK_IP_ADDR}:5000/detections/{detection.filename}"} for detection in detections]
    return jsonify(users_list)
    
@app.route('/detections/<string:filename>')
def get_image(filename):
    detection = Detection.query.filter_by(filename=filename).first()
    if not detection:
        detection = TempDetection.query.filter_by(filename=filename).first()
    
    return send_file(io.BytesIO(detection.image), mimetype=detection.mimetype, download_name=detection.filename)

@app.route('/')
def index():
    return "CAQAO Server"

def get_json_response(detection):
    json_response = {
        'id': detection.id,
        'img_src_url': f"http://{FLASK_IP_ADDR}:5000/detections/{detection.filename}",
        "veryDarkBrown" : detection.veryDarkBrown,
        "brown" : detection.brown,
        "partlyPurple" : detection.partlyPurple,
        "totalPurple" : detection.totalPurple,
        "g1" : detection.g1,
        "g2" : detection.g2,
        "g3" : detection.g3,
        "g4" : detection.g4,
        "mouldy" : detection.mouldy,
        "insectInfested" : detection.insectInfested,
        "slaty" : detection.slaty,
        "germinated" : detection.germinated,
        "beanGrade": detection.beanGrade
    }
    return jsonify(json_response)


def get_class_detection_counts(results):
    class_counts = {
        "veryDarkBrown" : 0,
        "brown" : 0,
        "partlyPurple" : 0,
        "totalPurple" : 0,
        "g1" : 0,
        "g2" : 0,
        "g3" : 0,
        "g4" : 0,
        "mouldy" : 0,
        "insectInfested" : 0,
        "slaty" : 0,
        "germinated" : 0,
    }

    for key, value in results.pandas().xyxy[0].name.value_counts().items():
        key = key.lower()
        key_split = key.split("-")
        if len(key_split) > 1:
            color, grade = key_split[0], key_split[1]
            color = color.split()
            if len(color) == 0:
                class_counts[color] += value
                class_counts[grade] += value
            else:
                color = color[0].lower() + ''.join(i.capitalize() for i in color[1:])
                class_counts[color] += value
                class_counts[grade] += value
        else:
            defect = key_split[0]
            defect = defect.split()
            if len(defect) == 0:
                class_counts[defect] += value
            else:
                defect = defect[0].lower() + ''.join(i.capitalize() for i in defect[1:])
                class_counts[defect] += value

    return class_counts

def get_bean_grade(class_count, bean_size):
    slaty, mouldy = class_count["slaty"], class_count["mouldy"]
    insect_infested, germinated = class_count["insectInfested"], class_count["germinated"]
    letter_code, num_code = "", ""
    tresholdNumCode = 0.03 * MAX_DET

    if (slaty <= tresholdNumCode) and (mouldy <= tresholdNumCode) and \
        ((insect_infested + germinated) <= tresholdNumCode):
            num_code = "1"
    else:
            num_code = "2"
    
    if bean_size <= 100:
        letter_code = "A"
    elif (bean_size >= 101) and (bean_size <= 110):
        letter_code = "B"
    else:
        letter_code = "C"
    
    return num_code + letter_code


if __name__ == "__main__":
    MAX_DET = 50
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")
    model.max_det = MAX_DET
    app.run(host=FLASK_IP_ADDR, port=5000, debug=True)
    # sample request
    # $ curl -X POST -F image=@test/sample.jpg 'http://192.168.72.32:5000/assess'