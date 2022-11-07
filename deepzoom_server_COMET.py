import scipy.io as sio
from flask import Flask, abort, make_response, render_template, url_for, request, jsonify
from io import BytesIO
import openslide
from openslide import ImageSlide, open_slide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import re
from unicodedata import normalize
import cv2
from PIL import Image
import numpy as np
from numpy.linalg import inv
from numpy.fft import fft2
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
import os

DEEPZOOM_SLIDE = None
DEEPZOOM_FORMAT = 'jpeg'
DEEPZOOM_TILE_SIZE = 256
DEEPZOOM_OVERLAP = 0
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 75
TARGET_SLIDE_NAME = 'slide1'
SOURCE_SLIDE_NAME = 'slide2'
DUPLICATE_TARGET_SLIDE_NAME = 'slide1_duplicate'

class LocalRegistration:
	def __init__(self, kernel=25, sigma=15):
		self.kernel = kernel
		self.sigma = sigma

	def get_mask_Hchannel(self, image):
		ihc_hed = rgb2hed(image)
		h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1), in_range='image')
		img = 255 - (255 * h)

		bimg = cv2.GaussianBlur(img, (self.kernel, self.kernel), self.sigma)
		ret, mask = cv2.threshold(bimg, np.mean(bimg), 255, cv2.THRESH_BINARY)
		mask = cv2.medianBlur(np.uint8(mask), 5)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
		complement = 255 - mask
		dseed = cv2.distanceTransform(np.uint8(complement), distanceType=cv2.DIST_L2, maskSize=5)
		_, mask, _, _ = cv2.floodFill(mask, None, np.unravel_index(dseed.argmax(), dseed.shape)[::-1], 0)
		mask = 255 - mask
		return mask

	def get_mask(self, image):
		ihc_hed = rgb2hed(image)
		h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1), in_range='image')
		d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1), in_range='image')
		img = 255 - (255 * (h + d) / 2)

		bimg = cv2.GaussianBlur(img, (self.kernel, self.kernel), self.sigma)
		ret, mask = cv2.threshold(bimg, np.mean(bimg), 255, cv2.THRESH_BINARY)
		mask = cv2.medianBlur(np.uint8(mask), 5)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
		complement = 255 - mask
		dseed = cv2.distanceTransform(np.uint8(complement), distanceType=cv2.DIST_L2, maskSize=5)
		_, mask, _, _ = cv2.floodFill(mask, None, np.unravel_index(dseed.argmax(), dseed.shape)[::-1], 0)
		mask = 255 - mask
		return mask

	def phaseCorrelation(self, Image1, Image2):
		fftI1 = fft2(Image1)
		fftI2 = fft2(Image2)

		F = fftI1 * fftI2.conjugate()
		Fn = F
		Fn[Fn != 0] = Fn[Fn != 0] / abs(Fn[Fn != 0])
		ir = np.fft.ifft2(F).real
		# F /= np.absolute(F)
		# ir = np.fft.ifft2(F).real
		return ir

	def pcOffset(self, reg, ref):
		pcImage = self.phaseCorrelation(reg, ref)

		filterSize = (5, 5)
		a = pcImage[-filterSize[0]:, :]
		b = pcImage[0:filterSize[1], :]
		pcImage = np.concatenate((a, pcImage, b), axis=0)
		a = pcImage[:, -filterSize[0]:]
		b = pcImage[:, 0:filterSize[1]]
		pcImage = np.concatenate((a, pcImage, b), axis=1)

		windowSize = (25, 25)
		pcImage = cv2.GaussianBlur(pcImage, windowSize, 4)
		pcImage = pcImage[filterSize[0]:-filterSize[0], filterSize[1]: -filterSize[1]]

		pcBW = pcImage >= np.percentile(pcImage, 99)

		label_image = label(pcBW)
		regions = regionprops(label_image)
		ccSizes = [e.area for e in regions]
		biggestI = np.argmax(ccSizes)
		biggest = ccSizes[biggestI]
		ccSizes = np.array(ccSizes)
		secondBiggest = np.amax(ccSizes[ccSizes < biggest])
		if secondBiggest == []:
			secondBiggest = 0

		offset = [0,0]
		pcImage_shape = np.array(pcImage.shape)
		if secondBiggest / biggest < 0.3:
			biggest_coord = regions[biggestI].coords
			offset = np.column_stack(np.where(pcImage == np.amax(pcImage[biggest_coord[:, 0], biggest_coord[:, 1]])))[0]
			offset = ((offset + pcImage_shape / 2) % pcImage_shape) - (pcImage_shape / 2) - 1
		else:
			centre = [int(i) for i in np.floor(pcImage_shape / 2)]
			a = pcImage[centre[0]:, centre[1]:]
			b = pcImage[centre[0]:, :centre[1]]
			top = np.concatenate((a, b), axis=1)
			a = pcImage[:centre[0], centre[1]:]
			b = pcImage[:centre[0], :centre[1]]
			bottom = np.concatenate((a, b), axis=1)
			pcImageA = np.concatenate((top, bottom), axis=0)

			pcBW = pcImageA >= np.percentile(pcImageA, 99)
			label_image = label(pcBW)
			regions = regionprops(label_image)
			ccSizes = [e.area for e in regions]
			biggestI = np.argmax(ccSizes)
			biggest = ccSizes[biggestI]
			ccSizes = np.array(ccSizes)
			secondBiggest = np.amax(ccSizes[ccSizes < biggest])
			if secondBiggest == []:
				secondBiggest = 0

			if secondBiggest / biggest < 0.2:
				biggest_coord = regions[biggestI].coords
				offset = \
				np.column_stack(np.where(pcImageA == np.amax(pcImageA[biggest_coord[:, 0], biggest_coord[:, 1]])))[0]
				offset = offset - (pcImage_shape / 2) - 1
				offset = ((offset + pcImage_shape / 2) % pcImage_shape) - (pcImage_shape / 2) - 1
		translation = np.float32([[1, 0, -offset[1]], [0, 1, -offset[0]], [0,0,1]])
		return translation


app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_TILER_SETTINGS', silent=True)
localRegObj = LocalRegistration()


class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')


@app.before_first_request
def load_slide():
    slidefile1 = app.config['DEEPZOOM_TARGET_SLIDE']
    slidefile2 = app.config['DEEPZOOM_SOURCE_SLIDE']
    app.GlobalTransform = app.config['TRANSFORM_MATRIX']
    app.LocalTransform = np.identity(3)
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


@app.route('/localReg', methods=['GET', 'POST'])
def localRegFtn():
    if request.method == 'POST':
        tile_info = request.get_json()
        if tile_info != '':
            all_paths = tile_info.split(";")
            all_levels = [int(path.split("/")[2]) for path in all_paths if path != '']
            all_cols = [int(path.split("/")[-1].split('.')[0].split('_')[0]) for path in all_paths if path != '']
            all_rows = [int(path.split("/")[-1].split('.')[0].split('_')[1]) for path in all_paths if path != '']

            unique_levels = np.unique(all_levels)
            if len(unique_levels)>1:
                print('tile are from more than one level!')

            iLevel = np.min(all_levels)
            indexes = [index for index in range(len(all_levels)) if all_levels[index] == iLevel]
            col_level = [all_cols[index] for index in indexes]
            row_level = [all_rows[index] for index in indexes]
            num_col_tiles = len(np.unique(col_level))
            num_row_tiles = int(len(col_level) / num_col_tiles)

            left_image = Image.new("RGB", (num_col_tiles * 256, num_row_tiles * 256))
            right_image = Image.new("RGB", (num_col_tiles * 256, num_row_tiles * 256))
            iCol, iRow = 0, 0
            for iter in range(len(col_level)):
                tile = app.slides['slide1'].get_tile(iLevel, (col_level[iter], row_level[iter]))
                left_image.paste(tile, (iCol * 256, iRow * 256))
                tile = get_transformed_tile(iLevel, (col_level[iter], row_level[iter]), 0)
                right_image.paste(tile, (iCol * 256, iRow * 256))
                iRow += 1
                if (iter + 1) % num_row_tiles == 0:
                    iCol, iRow = iCol + 1, 0
                # print('%d: %d'%(iCol, iRow))

            targetImage = np.asarray(np.copy(left_image))
            sourceImage = np.asarray(np.copy(right_image))
            targetImage[(targetImage[:, :, 0] == 0) & (targetImage[:, :, 1] == 0) & (targetImage[:, :, 2] == 0)] = [243, 243, 243]
            sourceImage[(sourceImage[:, :, 0] == 0) & (sourceImage[:, :, 1] == 0) & (sourceImage[:, :, 2] == 0)] = [243, 243, 243]

            # targetMask = localRegObj.get_mask(targetImage)
            # sourceMask = localRegObj.get_mask(sourceImage)
            targetMask = localRegObj.get_mask_Hchannel(targetImage)
            sourceMask = localRegObj.get_mask_Hchannel(sourceImage)

            localT = localRegObj.pcOffset(sourceMask, targetMask)
            print(localT)

            # registered = cv2.warpAffine(sourceImage, localT[0:-1][:], targetMask.shape[:2][::-1])
            # plt.subplot(221)
            # plt.imshow(targetImage)
            # plt.subplot(222)
            # plt.imshow(registered)
            # plt.subplot(223)
            # plt.imshow(targetMask)
            # plt.subplot(224)
            # plt.imshow(sourceMask)
            # plt.show()

            if not np.array_equal(localT, np.identity(3)):
                scaling_factor = app.slides['slide2'].level_count - iLevel - 1  # rescale the translation depending on the current scale
                scaleTranslation = [[1, 1, (2 ** scaling_factor)], [1, 1, (2 ** scaling_factor)], [0, 0, 1]]
                app.LocalTransform = localT * scaleTranslation
                print(app.LocalTransform)

        return 'OK', 200

@app.route('/')
def index():
    slide_url1 = url_for('dzi', slug=TARGET_SLIDE_NAME)
    slide_url2 = url_for('dzi', slug=DUPLICATE_TARGET_SLIDE_NAME)
    return render_template('slide-multipane.html', slide_url1=slide_url1, slide_url2=slide_url2, slide_mpp=app.slide_mpp)


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
            # tile = Image.new('RGB', (DEEPZOOM_TILE_SIZE, DEEPZOOM_TILE_SIZE))
            # tile.paste(app.slides['slide1'].get_tile(level, (col, row)), (1,1))
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
    # target_tile = Image.new('RGB', (DEEPZOOM_TILE_SIZE, DEEPZOOM_TILE_SIZE))
    # target_tile.paste(app.slides['slide1'].get_tile(level, target_tile_address), (1,1))
    target_tile = app.slides['slide1'].get_tile(level, target_tile_address)
    transformSize = target_tile.size
    tileSize = DEEPZOOM_TILE_SIZE + (2*DEEPZOOM_OVERLAP)

    scaling_factor = app.slides['slide2'].level_count - level - 1                   # rescale the translation depending on the current scale
    scaleTranslation = [[1, 1, 1/(2 ** scaling_factor)], [1, 1, 1/(2 ** scaling_factor)], [0, 0, 1]]
    if isLocal:
        level_transform = np.matmul(app.LocalTransform, app.GlobalTransform) * scaleTranslation
    else:
        level_transform = app.GlobalTransform * scaleTranslation

    target_tile_coor = np.array(target_tile_address) * tileSize
    target_tile_dimesion = np.array(transformSize)
    target_centre = target_tile_coor + (target_tile_dimesion/2)
    target_centre = np.expand_dims(target_centre, axis=0)
    inv_transform_matrix = inv(level_transform)
    source_centre = transform_points(target_centre, inv_transform_matrix)[0]
    source_tile_address = [np.floor(x/tileSize) for x in source_centre]
    source_tile_coor = np.array(source_tile_address) * tileSize

    xTiles, yTiles = [-1, 0, 1], [-1, 0, 1]
    numTiles = app.slides['slide2'].level_tiles[level]

    if source_tile_address[0] == 0:
        xTiles.remove(-1)
    elif source_tile_address[0] < 0:
        xTiles.remove(0)
        xTiles.remove(-1)
    if source_tile_address[0] >= numTiles[0]-1:
        xTiles.remove(1)
    if source_tile_address[1] == 0:
        yTiles.remove(-1)
    elif source_tile_address[1] < 0:
        yTiles.remove(0)
        yTiles.remove(-1)
    if source_tile_address[1] >= numTiles[1]-1:
        yTiles.remove(1)

    image = Image.new('RGB', (tileSize*3, tileSize*3))
    for ix in xTiles:
        for iy in yTiles:
            # print('%d:%d \n' % (ix, iy))
            if source_tile_address[0] + ix > numTiles[0] - 1 or source_tile_address[1] + iy > numTiles[1] - 1:
                img = Image.new('RGB', (tileSize, tileSize))
            else:
                img = app.slides['slide2'].get_tile(level, (source_tile_address[0] + ix, source_tile_address[1] + iy))
            image.paste(img, ((ix + 1) * tileSize, (iy + 1) * tileSize))

    # perform transformation and then crop the centre part
    offset_tile_centre = (source_tile_coor + tileSize/2) - source_centre
    translateT = np.float32([[1,0,offset_tile_centre[0]],[0,1,offset_tile_centre[1]],[0,0,1]])
    tempT = level_transform * [[1, 1, 0], [1, 1, 0], [1, 1, 1]]  # remove translation
    Translation_ = np.array([[1, 0, -image.size[0] / 2], [0, 1, -image.size[1] / 2], [0, 0, 1]])
    Translation = np.array([[1, 0, image.size[0] / 2], [0, 1, image.size[1] / 2], [0, 0, 1]])
    tempT = np.matmul(np.matmul(np.matmul(Translation, tempT), Translation_), translateT)
    transform_image = cv2.warpAffine(np.array(image), tempT[0:-1][:], image.size[:2])
    image_centre = (int(tileSize*1.5), int(tileSize*1.5))
    temp = transform_image[image_centre[1] - np.floor(transformSize[1] / 2).astype(int):image_centre[1] + np.ceil(transformSize[1] / 2).astype(int),
           image_centre[0] - np.floor(transformSize[0] / 2).astype(int):image_centre[0] + np.ceil(transformSize[0] / 2).astype(int), :]

    image = Image.fromarray(temp)
    return image


def transform_points(points, matrix):
    """ transform points according to given transformation matrix

    :param ndarray points: set of points of shape (N, 2)
    :param ndarray matrix: transformation matrix of shape (3, 3)
    :return ndarray: warped points  of shape (N, 2)
    """
    points = np.array(points)
    # Pad the data with ones, so that our transformation can do translations
    pts_pad = np.hstack([points, np.ones((points.shape[0], 1))])
    points_warp = np.dot(pts_pad, matrix.T)
    return points_warp[:, :-1]


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
        app.config['DEEPZOOM_TARGET_SLIDE'] = args[0]
        app.config['DEEPZOOM_SOURCE_SLIDE'] = args[1]

    except IndexError:
        if app.config['DEEPZOOM_TARGET_SLIDE'] is None or app.config['DEEPZOOM_SOURCE_SLIDE'] is None:
            parser.error('Please provide both target and source slides!')
    try:
        transform_path = args[2]
        app.config['TRANSFORM_MATRIX'] = np.load(transform_path)

    except IndexError:
        parser.error('The given transform file is not in a correct format!')

    app.run(host=opts.host, port=opts.port, threaded=True, debug=True)
