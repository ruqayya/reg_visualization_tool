from flask import Flask, abort, make_response, render_template, url_for, request, jsonify
from io import BytesIO
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
from unicodedata import normalize
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

DEEPZOOM_SLIDE = None
DEEPZOOM_FORMAT = 'jpeg'
DEEPZOOM_TILE_SIZE = 256
DEEPZOOM_OVERLAP = 0
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 75
TARGET_SLIDE_NAME = 'slide1'
SOURCE_SLIDE_NAME = 'slide2'
DUPLICATE_TARGET_SLIDE_NAME = 'slide1_duplicate'

app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_TILER_SETTINGS', silent=True)

class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')

@app.before_first_request
def load_slide():
    slidefile1 = app.config['DEEPZOOM_TARGET_SLIDE']
    slidefile2 = app.config['DEEPZOOM_SOURCE_SLIDE']

    if slidefile1 is None or slidefile2 is None:
        raise ValueError('Please specify both target and source image.')
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = dict((v, app.config[k]) for k, v in config_map.items())
    slide1 = open_slide(slidefile1)
    slide2 = open_slide(slidefile2)
    app.slides = {
        TARGET_SLIDE_NAME: DeepZoomGenerator(slide1, **opts),
        SOURCE_SLIDE_NAME: DeepZoomGenerator(slide2, **opts),
        DUPLICATE_TARGET_SLIDE_NAME: DeepZoomGenerator(slide1, **opts)
    }
    try:
        mpp_x = slide1.properties[openslide.PROPERTY_NAME_MPP_X]
        mpp_y = slide1.properties[openslide.PROPERTY_NAME_MPP_Y]
        app.slide_mpp = (float(mpp_x) + float(mpp_y)) / 2
    except (KeyError, ValueError):
        app.slide_mpp = 0

@app.route('/')
def index():
    slide_url1 = url_for('dzi', slug=TARGET_SLIDE_NAME)
    slide_url2 = url_for('dzi', slug=DUPLICATE_TARGET_SLIDE_NAME)
    return render_template('slide-multipane_histokat.html', slide_url1=slide_url1, slide_url2=slide_url2, slide_mpp=app.slide_mpp)

@app.route('/<slug>.dzi')
def dzi(slug):
    format = app.config['DEEPZOOM_FORMAT']
    try:
        resp = make_response(app.slides[slug].get_dzi(format))
        resp.mimetype = 'application/xml'
        return resp
    except KeyError:
        # Unknown slug
        abort(404)

@app.route('/<slug>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(slug, level, col, row, format):
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        if slug == 'slide1':
            tile = app.slides[slug].get_tile(level, (col, row))
        else:
            tile = get_transformed_tile(level, (col, row), 1)
    except KeyError:
        # Unknown slug
        abort(404)
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp

def get_transformed_tile(level, target_tile_address, isLocal = 1):
    moving_tile = app.slides['slide2'].get_tile(level, target_tile_address)
    return moving_tile

def slugify(text):
    text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]+', '-', text)

if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] [slide]')
    parser.add_option('-B', '--ignore-bounds', dest='DEEPZOOM_LIMIT_BOUNDS',
                default=True, action='store_false',
                help='display entire scan area')
    parser.add_option('-c', '--config', metavar='FILE', dest='config',
                help='config file')
    parser.add_option('-d', '--debug', dest='DEBUG', action='store_true',
                help='run in debugging mode (insecure)')
    parser.add_option('-e', '--overlap', metavar='PIXELS',
                dest='DEEPZOOM_OVERLAP', type='int',
                help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{jpeg|png}',
                dest='DEEPZOOM_FORMAT',
                help='image format for tiles [jpeg]')
    parser.add_option('-l', '--listen', metavar='ADDRESS', dest='host',
                default='127.0.0.1',
                help='address to listen on [127.0.0.1]')
    parser.add_option('-p', '--port', metavar='PORT', dest='port',
                type='int', default=5006,
                help='port to listen on [5000]')
    parser.add_option('-Q', '--quality', metavar='QUALITY',
                dest='DEEPZOOM_TILE_QUALITY', type='int',
                help='JPEG compression quality [75]')
    parser.add_option('-s', '--size', metavar='PIXELS',
                dest='DEEPZOOM_TILE_SIZE', type='int',
                help='tile size [254]')

    (opts, args) = parser.parse_args()

    # Load config file if specified
    if opts.config is not None:
        app.config.from_pyfile(opts.config)

    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith('_') and getattr(opts, k) is None:
            delattr(opts, k)
    app.config.from_object(opts)

    # Set slide file
    try:
        # app.config['DEEPZOOM_TARGET_SLIDE'] = args[0]
        # app.config['DEEPZOOM_SOURCE_SLIDE'] = args[1]
        app.config['DEEPZOOM_TARGET_SLIDE'] = "Histology_WSIs\\06-18270_5_A1CK818_2.tif"
        app.config['DEEPZOOM_SOURCE_SLIDE'] = "Histology_WSIs\\06-18270_5_CK818_2_MLH1.tif"
    except IndexError:
        if app.config['DEEPZOOM_TARGET_SLIDE'] is None or app.config['DEEPZOOM_SOURCE_SLIDE'] is None:
            parser.error('Please provide both target and source slides!')

    app.run(host=opts.host, port=opts.port, threaded=True, debug=True)
