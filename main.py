import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import pylab
from PIL import Image

rng = numpy.random.RandomState(23455)

# istantiate 4D tensor for input
input = T.tensor4(name='input')

# initialize shared variable for weights
w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
W = theano.shared(numpy.asarray(rng.uniform(
				low=-1.0/w_bound,
				high=1.0/w_bound,
				size=w_shp),
			dtype=input.dtype), name='W')

# initialize shared variable for bias
b_shp = (2,)
b = theano.shared(numpy.asarray(rng.uniform(
				low=-.5,
				high=.5,
				size=b_shp),
			dtype=input.dtype), name='b')

# build symbolic expression to compute the
# convolution of the input with filters in W
conv_out = conv.conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)

# open image and read it as an array of float64
img = Image.open(open('3wolfmoon.jpg'))			# image shape is 639x516
img = numpy.asarray(img, dtype='float64') / 256.	# divide by 256 to get RGB values

# put image in 4D tensor of shape (1, 3, height, width)
# which corresponds to (mini batch size, number of input 
# feature maps, image height, image width)
img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 639, 516)
filtered_img = f(img_)

# plot original and first and second components of output
pylab.subplot(131, title='Original'); pylab.axis('off'); pylab.imshow(img)
pylab.gray()
pylab.subplot(132, title='Convolved 1'); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(133, title='Convolved 2'); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()
