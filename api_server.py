from flask import Flask, request, send_file
# from flask.helpers import make_response
from werkzeug.utils import secure_filename

from stylegan2_ada_pytorch.projector import run_projection
import face_recognition


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
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image)
        print(face_locations)
        print(face_locations[0])
        print(face_locations[0][1])


        run_projection(
        network_pkl = './dnnlib/cel2/network-snapshot-000600.pkl',
        target_fname = filepath,
        outdir = './save/out',
        save_video = False,
        seed = 100,
        num_steps = 10
        )

        return send_file('./save/out/proj.png')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port='7000')