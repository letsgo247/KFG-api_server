from flask import Flask, request, send_file
# from flask.helpers import make_response
from werkzeug.utils import secure_filename

from stylegan2_ada_pytorch.projector import run_projection
import face_recognition
import cv2
import numpy as np


app = Flask(__name__)
# CORS(app)


@app.route('/test/<name>')
def success(name):
    return 'Welcome %s' % name


@app.route('/ganarate', methods=('GET', 'POST'))
def ganarate():
    if request.method == 'GET':
        # print('**********************')
        # print(request.args.to_dict())
        # print('**********************')
        # img = request.args['img']
        # print('=====================')
        # print(img.filename)
        # print('=====================')
        # # img.save(secure_filename(img.filename))
        
        '''
        GET: request.args
        POST: request.form
        '''

        return text


    elif request.method == 'POST':
        print('**********************')
        print(request.files)
        # print(request.form.to_dict())
        img = request.files['img']
        # print(img)
        print('=====================')
        filepath = './save/'+secure_filename(img.filename)

        img.save(filepath)
        
        detected_faces = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(detected_faces)
        print(face_locations)
        print(face_locations[0])    # 얼굴이 여러개라면, 첫번째 것만 선택?! (안내: 못찾는 경우, 여러개 찾는 경우)
        (y_top, x_right, y_bottom, x_left) = face_locations[0]
        width = x_right - x_left
        height = y_bottom - y_top
        margin = 0.4
        
        ff = np.fromfile(filepath, np.uint8)
        image = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
        h,w,_ = image.shape
        print('img width:',w)
        print('img height:',h)

        y_top_new = y_top - int(height*margin)
        if y_top_new<0:
            y_top_new = 0
        
        y_bottom_new = y_bottom + int(height*margin)
        if y_bottom_new>h:
            y_bottom_new = h
        
        x_left_new = x_left - int(width*margin)
        if x_left_new<0:
            x_left_new = 0
        
        x_right_new = x_right + int(width*margin)
        if x_right_new>w:
            x_right_new = w

        image_crop = image[y_top_new:y_bottom_new, x_left_new:x_right_new]
        cropped_filepath = './save/cropped_'+secure_filename(img.filename)
        cv2.imwrite(cropped_filepath, image_crop)



        run_projection(
        network_pkl = './dnnlib/cel2/network-snapshot-000600.pkl',
        target_fname = cropped_filepath,
        outdir = './save/out',
        save_video = True,
        seed = 100,
        num_steps = 100
        )

        return send_file('./save/out/proj.png')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port='7000')