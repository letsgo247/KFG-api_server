from flask import Flask, request, send_file
# from flask.helpers import make_response
from werkzeug.utils import secure_filename

from stylegan2_ada_pytorch.projector import run_projection

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
        print('=====================')
        # print(img.filename)
        filepath = './save/'+secure_filename(img.filename)

        img.save(filepath)

        run_projection(
        network_pkl = './dnnlib/network-snapshot-000800.pkl',
        target_fname = filepath,
        outdir = './save/out',
        save_video = False,
        seed = 100,
        num_steps = 10
        )

        return send_file('./save/out/proj.png')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.2', port='7000')