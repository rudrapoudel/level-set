"""
Local Otsu Threshold in python

Rudra Poudel
"""
__docformat__ = 'restructedtext en'

from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank

def otsuThreshold(img):
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu

    return global_otsu.astype(int)

def otsuThresholdLocal(img, radius = 15):
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)
    local_otsu = img >= local_otsu

    return local_otsu.astype(int)

if __name__ == '__main__':
    # -------------------------------------------
    import scipy, numpy
    from PIL import Image
    img = Image.open('image/house.jpg')
    img = numpy.asarray(img)

    print(numpy.min(img))
    print(numpy.max(img))
    print(img.dtype)

    log = 'log/'

    gray = otsuThresholdLocal(img.copy())
    gray = scipy.misc.toimage(gray, cmin = 0, cmax = 1)
    gray.save(log + 'otsu_local_out_py.png')

    gray = otsuThreshold(img.copy())
    gray = scipy.misc.toimage(gray, cmin = 0, cmax = 1)
    gray.save(log + 'otsu_out_py.png')
    # -------------------------------------------
