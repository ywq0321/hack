import os
import uuid
import subprocess
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename, safe_join

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MASKS_PARENT = os.path.join(BASE_DIR, "masks")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MASKS_PARENT, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff"}

app = Flask(__name__)
app.config["UPLOAD_DIR"] = UPLOAD_DIR
app.config["MASKS_PARENT"] = MASKS_PARENT


def allowed_file(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/segment', methods=['POST'])
def segment():
    # Receives image file and model path (text). Runs segmentation.py and returns masks list
    if 'image' not in request.files:
        return jsonify({'error': 'no image file'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'file type not allowed'}), 400

    model_path = request.form.get('model_path', '').strip()
    if model_path == '':
        # default model filename (user should change if necessary)
        model_path = 'yolov8s-seg.pt'

    uid = str(uuid.uuid4())
    saved_name = secure_filename(uid + '_' + file.filename)
    image_path = os.path.join(app.config['UPLOAD_DIR'], saved_name)
    file.save(image_path)

    # create output dir for masks
    out_dir = os.path.join(app.config['MASKS_PARENT'], uid)
    os.makedirs(out_dir, exist_ok=True)

    # call the segmentation script as a subprocess
    # segmentation.py must accept --image, --model, --output_dir
    cmd = ['python', 'segmentation.py', '--image', image_path, '--model', model_path, '--output_dir', out_dir]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        return jsonify({'error': f'error invoking segmentation script: {e}'}), 500

    # return stdout/stderr for debugging
    stdout = completed.stdout
    stderr = completed.stderr

    # collect mask files
    masks = []
    for fname in sorted(os.listdir(out_dir)):
        if fname.lower().endswith('.png'):
            masks.append({'name': fname, 'url': f'/masks/{uid}/{fname}'})

    return jsonify({'masks': masks, 'stdout': stdout, 'stderr': stderr, 'image_url': f'/uploads/{saved_name}'})


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_DIR'], filename)


@app.route('/masks/<uid>/<path:filename>')
def mask_file(uid, filename):
    # safe join
    dirpath = safe_join(app.config['MASKS_PARENT'], uid)
    return send_from_directory(dirpath, filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)