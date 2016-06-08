require 'torch'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local img, gray
img = image.load('house.jpg')[1]
img:mul(255):round()

local py = require('fb.python')
py.exec([=[
import numpy as np
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank

def otsuThreshold(img):
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu

    return global_otsu.astype(int)

def otsuThresholdLocal(img, radius = 15):
    img = np.uint8(img)
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)
    local_otsu = img >= local_otsu

    return local_otsu.astype(int)
]=] ) 


-- TEST
log = ''
-- log = '/home/rp14/log/'
-- log = '/Users/rudra/log/'

gray = py.eval('otsuThresholdLocal(img)', {img = img})
image.save(log .. 'otsu_local_out_py_lua.png', gray)

gray = py.eval('otsuThreshold(img)', {img = img})
image.save(log .. 'otsu_out_py_lua.png', gray)
