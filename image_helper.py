###################################################################
# Imports
# sci-kit image
from skimage import exposure
from skimage.color import rgb2gray
from skimage import color
from skimage.util import crop
from skimage.transform import rotate
from skimage import color
from skimage.measure import profile_line
from scipy import ndimage, misc
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy import fftpack
# Movies
try:
    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.all import crop
except:
    print('moviepy not installed, cannot do movie operations')
# Scalebar
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
# misc.
from matplotlib.colors import LinearSegmentedColormap
try:
    import PIL.Image
    import PIL.ExifTags
except:
    print('PIL not installed. ExifTags cannot be handled')
import numpy as np
###################################################################

    
def getScalingFactors(img,origin,p1,p2,r1,r2):
    '''Function to rescale axes, given calibration points
    origin = tuple of Y,X indices of origin
    p1,p2 = tuples of Y,X of two calibration points
    r1,r2 = distance from origin of calibration points'''
    # First, specify origin
    # Get length of each dimension
    if img.ndim==2: # grayscale image
        ylen, xlen = img.shape
    elif img.ndim==3: # ignore color channel
        ylen,xlen,_ = img.shape
    # Make x and y arrays
    X = np.arange(0,xlen,1)
    Y = np.arange(0,ylen,1)
    X = X-X[origin[0]]
    Y = Y-Y[origin[1]]
    x1 = X[p1[0]]
    y1 = Y[p1[1]]
    x2 = X[p2[0]]
    y2 = Y[p2[1]]
    b = np.sqrt( (r2**2 - r1**2/x1**2) / (y2**2 - y1**2 * x2**2 / x1**2) )
    a = np.sqrt( (r1**2 - b**2 * y1**2)/x1**2 )
    return a,b

def getScaledAxes(img,origin,a,b):
    # Get length of each dimension
    ylen, xlen = img.shape

    # Make x and y arrays
    X = np.arange(0,xlen,1)#np.linspace(-5,5,num=xlen)
    Y = np.arange(0,ylen,1)#np.linspace(-5,5,num=ylen)
    
    # Align origin and rescale to match distance
    X = X-X[origin[0]]
    Y = Y-Y[origin[1]]
    X = X*a
    Y = Y*b
    return(X,Y)

def plotMesh(img,origin,a,b,cmap,ax,int_cut_off,rad_cut_off=None):
    # int_cut_off is lower intensity threshold for plotting
    # rad_cut_off is lower radius threshold for plotting
    X,Y = getScaledAxes(img,origin,a,b)
    # select only pixels above certain value
    maskedImg = np.ma.masked_less(img, int_cut_off)
    if rad_cut_off:
        y, x = np.indices((img.shape))
        r = np.sqrt(((x-origin[0])*a)**2 + ((y-origin[1])*b)**2)
        maskedImg = np.ma.masked_where(r<rad_cut_off,maskedImg)
    # Rotate image?
    # Plot
    ax.pcolormesh(X,Y,maskedImg,cmap=cmap,vmin=0,alpha=0.3)

def radial_profile(img,origin,a,b,factor=300):
    '''
    a is length/pixel conversion for x
    b is length/pixel conversion for y
    origin is (x,y) tuple of the origin
    factor defines how many bins are integrated over
    because the integration method uses integer bins
    '''
    y, x = np.indices((img.shape))
    r = np.sqrt(((x-origin[0])*a)**2 + ((y-origin[1])*b)**2)*factor
    r = r.ravel().astype(np.int)
    tbin = np.bincount(r, img.ravel())
    nr = np.bincount(r)
    radialprofile = tbin / nr
    r=np.indices((radialprofile.shape))[0]/factor
    return r,radialprofile

def getExposureTime(image_file):
    img = PIL.Image.open(image_file)
    exif = {
    PIL.ExifTags.TAGS[k]: v
    for k, v in img._getexif().items()
    if k in PIL.ExifTags.TAGS
    }
    # 'ExposureTime' has form (numerator,denominator)
    # Return as decimal in seconds
    exposure_time = exif['ExposureTime'][0]/exif['ExposureTime'][1]
    return exposure_time
    

# Calibration values for Nikon microscope
micron_per_pixel = {'4x':1000/696, '10x':1000/1750,
                       '20x':500/1740, '50x':230/2016}
def add_scalebar(ax,length,unit,mag,length_per_pixel=None,
                 show_label=True,label_top=True,height=20,
                 loc='lower right',color='white',fontsize=16):
    '''
    ax = axis handle
    length = length of scalebar in 'unit'
    unit = unit of length, mm for millimeter or um for microns
    mag = magnification of microscope, '4x','10x','20x',or '50x'
    length_per_pixel = conversion from pixel to length
        default is None, using calibration factors for the Nikon
    height = height of scalebar
    loc = location specifier of scalebar
    '''
    if unit == 'um':
        label = str(length)+' $\mu$m'
        factor = 1
    if unit == 'mm':
        label = str(length)+' mm'
        factor = 1e-3
    # calibration distances for the Nikon microscope
    micron_per_pixel = {'4x':1000/696, '10x':1000/1750,
                       '20x':500/1740, '50x':230/2016}
    if not length_per_pixel:
        length_per_pixel = micron_per_pixel[mag]
    pixels = length / (length_per_pixel * factor)
    if not show_label:
        label = ""
    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(ax.transData,
                               pixels, label, loc, 
                               pad=0.1,
                               color=color,
                               frameon=False,
                               size_vertical=height,
                               label_top=label_top,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)
    return ax

def burn_scalebar(img, mag=None, micron_per_pixel=None, 
                    width_microns=20, height_microns=4, color='white',
                    location='lower left'):
    '''
    This function burns a scalebar into a numpy array image
        this allows a scalebar to be added without matplotlib
    img is 2d or 3d numpy array (gray or color image)
    mag is image magnification, 
        only supply if taken by lumenera software on nikon microscope
        if None, then must provide micron_per_pixel
    color can be 'white','black', value between 0 and 1, or [r,g,b]
        e.g. white = [255,255,255] for color image
    '''
    if mag:
        # calibration distances for the Nikon microscope
        micron_per_pixel_dict = {'4x':1000/696, '10x':1000/1750,
                           '20x':500/1740, '50x':230/2016}
        micron_per_pixel=micron_per_pixel_dict[mag]
    else:
        if not micron_per_pixel:
            raise Exception('Must supply magnification or micron_per_pixel')
    # Get img dimensions
    img_width = img.shape[0]
    img_height = img.shape[1]
     
    # Burn the scale bar by changing pixel values
    width = width_microns / micron_per_pixel # 20 microns wide
    height = height_microns / micron_per_pixel
    if location == 'lower right':
        y1 = int(img_height * 0.95)
        x1 = int(img_width * 0.95)
    elif location == 'lower left':
        y1 = int(img_height * 0.95)
        x1 = int(img_width * 0.05 + width)
    elif location == 'upper right':
        y1 = int(img_height * 0.05 + height)
        x1 = int(img_width * 0.95)
    elif location == 'upper left':
        y1 = int(img_height * 0.05 + height)
        x1 = int(img_width * 0.05 + width)
        
    if img.ndim==3: # Color image
        if type(color) is str:
            assert color=='white' or color=='black', "color must be 'white' or 'black'"
            color={'white':[255,255,255],'black':[0,0,0]}[color]
        elif (type(color) is float) or (type(color) is int):
            assert color <=1 and color>=0, 'color must be between 0 and 1'
            color=color*np.array([255,255,255])
        else:
            assert (type(color) is list or type(color) is np.ndarray) and len(color)==3, \
                    'color must be list or numpy array with length=3' 
        img[(y1-height):y1,(x1-width):x1] = color
    elif img.ndim==2:
        if type(color) is str:
            assert color=='white' or color=='black', "color must be 'white' or 'black'"
            color = {'white':1,'black':0}[color]
        elif (type(color) is float) or (type(color) is int):
            assert color <=1 and color>=0, 'color must be between 0 and 1'
        img[(y1-height):y1,(x1-width):x1] = color
    return img
    
    
# Useful contrast methods from skimage
#exposure.equalize_adapthist(img, kernel_size=None, clip_limit=0.01)
#exposure.equalize_hist(img)
#exposure.rescale_intensity(image, in_range='image')
def distanceFromMeanContrast(img,below,above):
    mean_intensity = np.mean(rgb2gray(img))*255
    v_min = mean_intensity - below
    v_max = mean_intensity + above
    return exposure.rescale_intensity(img, in_range=(v_min, v_max))

# Kyle wrote the original version of this function
def constantMeanContrast(img,level):
    ar = np.array(img)
    # RGB image
    if len(np.shape(ar))==3:
        # Find deviation from mean intensity in red, green, and blue
        r=ar[:,:,0]
        mean_r=np.mean(r)
        diff_r=r-mean_r
        g=ar[:,:,1]
        mean_g=np.mean(g)
        diff_g=g-mean_g
        b=ar[:,:,2]
        mean_b=np.mean(b)
        diff_b=b-mean_b
        # Multiply difference from mean by 'level'
        ar[:,:,0]=mean_r+diff_r*level
        ar[:,:,1]=mean_g+diff_g*level
        ar[:,:,2]=mean_b+diff_b*level
        ar[ar>255]=255
        ar[ar<0]=0
        # Turn pixels that are saturated in one color into white or black pixels
        # not sure how to do this yet
    # Gray scale image
    elif len(np.shape(ar))==2:
        mean_int = np.mean(ar)
        difference = ar - mean_int
        ar = mean_int + difference*level
        ar[ar>255]=255
        ar[ar<0]=0
    
    # Turn pixels that are saturated in one color into white or black pixels
    #x,y,z=np.where(ar[...,:3]==255)
    #for i,e in enumerate(x):
    #    ar[x[i],y[i],:] = 255
    #x,y,z=np.where(ar==0)
    #for i,e in enumerate(x):
    #    ar[x[i],y[i],:] = 0
    #    ar[x[i],y[i],3] = 255
    
    return ar


def colorize(image, hue, saturation=1):
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)
    
##### Line profile measurements

def rotate_line(line,rot):
    # endpoint remains fixed
    theta = rot * np.pi / 180
    #rotation matrix
    R = np.array([[np.cos(theta),-np.sin(theta)],
                 [np.sin(theta),np.cos(theta)]])
    line_rot=[0,0]
    line_rot[0] = np.dot(R,line[0])
    line_rot[1] = np.dot(R,line[1])
    offset = line[1]-line_rot[1]
    line_rot = np.array(line_rot)+np.array(offset)
    return line_rot
def line_offset(line,xy_offset):
    return np.array(line)+np.array(xy_offset) 
def _plot_line(ax,line,rot=0,color='r'):
    '''
    ax is axis handle for plot
    rot is rotation angle in degrees
    line is list of tuples of start and endpoint
       e.g. [(x1,y1),(x2,y2)]
    end point stays fixed
    '''
    #rotation matrix
    theta = rot * np.pi / 180
    R = np.array([[np.cos(theta),-np.sin(theta)],
                 [np.sin(theta),np.cos(theta)]])
    line_rot=[0,0]
    line_rot[0] = np.dot(R,line[0])
    line_rot[1] = np.dot(R,line[1])
    offset = line[1]-line_rot[1]
    line_rot = np.array(line_rot)+np.array(offset)
    x,y = zip(*line_rot)
    ax.plot(x,y,color=color)
    return line_rot
def get_line_length(line,mag,unit='um',length_per_pixel=None):
    '''
    ax = axis handle
    length = length of scalebar in 'unit'
    unit = unit of length, mm for millimeter or um for microns
    mag = magnification of microscope, '4x','10x','20x',or '50x'
    length_per_pixel = conversion from pixel to length
        default is None, using calibration factors for the Nikon
    height = height of scalebar
    loc = location specifier of scalebar
    '''
    if unit == 'um':
        factor = 1
    if unit == 'mm':
        factor = 1e-3
    # calibration distances for the Nikon microscope
    micron_per_pixel = {'4x':1000/696, '10x':1000/1750,
                       '20x':500/1740, '50x':230/2016}
    if not length_per_pixel:
        length_per_pixel = micron_per_pixel[mag]
    x,y = zip(*line)
    pixels = np.sqrt( (x[1]-x[0])**2 + (y[1]-y[0])**2 )
    length = pixels * length_per_pixel * factor
    return length
    
def line_profile_dspacing(image,line,mag,unit='um',length_per_pixel=None):
    '''
    line = [(x1,y1),(x2,y2)] 
    mag = '4x','10x','20x', or '50x'
    '''
    # input to profile needs to be (y,x) or (row,col)
    # Opposite convention as axes in imshow
    profile = profile_line(rgb2gray(image),
                           (line[0][1],line[0][0]),
                           (line[1][1],line[1][0]))
    line_length= get_line_length(line,mag,unit='um',
                                 length_per_pixel=length_per_pixel)
    x = np.indices(np.shape(profile))[0]
    x = x/x[-1] * line_length
    # find peaks
    peak_idx = argrelextrema(profile,np.greater,order=2)[0] 
    # divide length/num peaks to get d-spacing
    d = (x[peak_idx[-1]]-x[peak_idx[0]])/(len(peak_idx)-1)
    return d,profile,x,peak_idx
    
##### FFT analysis
def radial_profile_fft(img,scale,factor=300):
    '''
    scale is length/pixel conversion for x
    factor defines how many bins are integrated over
    because the integration method uses integer bins
    '''
    # Get frequency wavenumbers based on image size
    # Then shift so that center of image is 0 frequency
    # I don't totally understand this, but it works
    ky = np.fft.fftshift(np.fft.fftfreq(img.shape[0], 
                                        scale)) 
    kx = np.fft.fftshift(np.fft.fftfreq(img.shape[1], 
                                            scale))
    kx=kx*np.ones(img.shape)
    #[:,None] changes shape from (N,) to (N,1)
    #identical to ky.reshape((ky.shape[0],1))
    ky=ky[:,None]*np.ones(img.shape)
    k = np.sqrt(kx**2 + ky**2)*factor
    k=k.ravel().astype(np.int)
    # Average azimuthally
    tbin = np.bincount(k, img.ravel())
    nr = np.bincount(k)
    radialprofile = tbin / nr
    k=np.indices((radialprofile.shape))[0]/factor
    # return arrays without region near origin
    s_idx = 10#np.where(1/k < 10)[0][0]
    return k[s_idx:],radialprofile[s_idx:]

def d_from_fft(image,scale=micron_per_pixel['50x'],order=20,factor=100,
              d_lower=0.7,d_upper=1.8):
    gray_image = rgb2gray(image)
    # Rescale to 0-255 (instead of 0-1)
    gray_image = ((gray_image - np.min(gray_image))/
                (np.max(gray_image) - np.min(gray_image)))
    # Take FFT and shift so 0 frequency is at center
    fft_image=fftpack.fft2(gray_image)
    fft_image=fftpack.fftshift(fft_image)
    # Take power spectral density
    # Not sure if squared is correct or not
    power2D = np.abs(fft_image)**2
    # Get azimuthally averaged radial profile of k=1/d
    k,radial=radial_profile_fft(power2D,scale,factor=factor)
    # Find local maxima
    # order could possibly be reduced to 10
    # background subtract might be useful
    peak_idx = argrelextrema(radial,np.greater,order=order)[0]
    # Take d only within reasonable range
    # This could be more robust
    d_peaks = 1/k[peak_idx]
    d_peak = d_peaks[(d_peaks>d_lower) & (d_peaks<d_upper)]
    if len(d_peak>=1):
        return {'d_peak':d_peak[0],'k':k,'radial':radial,'peak_idx':peak_idx}
    else:
        return {'d_peak':-1,'k':k,'radial':radial,'peak_idx':peak_idx}