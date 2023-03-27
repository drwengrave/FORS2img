description="quick reduction of fors imaging"
import os,sys,time,glob
import numpy as np
from astropy.io import fits
from ccdproc import cosmicray_lacosmic
from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image
from photutils.detection import DAOStarFinder
from scipy.ndimage import shift
from astropy.nddata import CCDData
from ccdproc import transform_image, combine
from astropy import units as u
import imexam
import subprocess
import argparse
import logging

start_time = time.time()
parser = argparse.ArgumentParser(description=description,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("imglist",help="img to process (with path, allowing \*) ")
parser.add_argument("-c","--caldir",help="master calibration directory",
                    default='caldir/')
parser.add_argument("-l","--list",dest="list",action="store_true",
                    default=False,help='list only')
parser.add_argument("--chip",dest='chip',help="chip to process",default='2')
parser.add_argument("--nocosmics",dest="nocosmics",action="store_true",
                    default=False,help='suppress cosmic correction')
parser.add_argument("--debug",dest="debug",action="store_true",
                    default=False,help='debug logging level')
args = parser.parse_args()   

level ="INFO"
if args.debug: level="DEBUG"
loglevel = getattr(logging,level)
logging.basicConfig(format='%(levelname)s:%(message)s',level=loglevel)
logger = logging.getLogger(__name__)

useful_keys = {'object'  :'OBJECT',
            'date'    :'DATE-OBS', 
            'mjd'     :'MJD-OBS',
            'RA'      :'RA',
            'DEC'     :'DEC',
            'exptime' :'EXPTIME',
            'extname': 'EXTNAME',
            'filter'  :'HIERARCH ESO INS FILT1 NAME',
            'imagetyp':  'HIERARCH ESO DPR TYPE',
            'obs_mode':'HIERARCH ESO DPR TECH',
            'imagecat':  'HIERARCH ESO DPR CATG',
            'seeing'  :'SEEING'}

trim = {"1": '[191:1850,3:950]', "2": '[195:1860,315:958]'}
saturat = 62000
epadu,ron = 0.8,3.6
scale = 0.251

def read_desc(img): ####################################

    imgkeys = {}
    try: hdr = fits.getheader(img)
    except:
       logger.error(f"error reading header for {img}")
       return imgkeys

    for d in useful_keys: 
        if useful_keys[d] in hdr: imgkeys[d] = hdr[useful_keys[d]]
        else: imgkeys[d] = 'NONE'

    return imgkeys

def group_data(_list):      

    _imglist = args.imglist
    filelist = glob.glob(f"{_imglist}")
    imglist = {}
    group = {}

    for im in sorted(filelist):
        
        _im = os.path.basename(im)[:-5]
        imgkeys = read_desc(im)
        if not imgkeys: continue
        imgkeys['filepath'] = im
        
        if imgkeys['imagetyp']!='OBJECT' or imgkeys['obs_mode']!='IMAGE' or \
            imgkeys['imagecat']!='SCIENCE' or \
            imgkeys['extname']!= f"CHIP{args.chip}":
            continue
        
        imglist[_im] = imgkeys

        f = imglist[_im]['filter']
        mjd = imglist[_im]['mjd']
        if f not in group: group[f] = {}
        if len(group[f].keys())==0: group[f][mjd] = [_im]
        else:
            mjdlist = np.array(list(group[f].keys()))
            mjdist = np.abs(mjdlist-mjd)
            ii = np.argmin(mjdist)
            if mjdist[ii]<.5: group[f][mjdlist[ii]].append(_im)
            else: group[f][mjd] = [_im]
            
    if _list:
        for f in group:
            print(f'filter {f}')
            for mjd in group[f]:
                print(f'mjd {mjd}')
                for im in group[f][mjd]:
                    print(f"{im} {imglist[im]['object']} {imglist[im]['filter']} {imglist[im]['date'][:10]}")
            
    return imglist,group

def dfile(filestring):
   
   trash = []
   for f in filestring.split(','): trash += glob.glob(f)
   for t in trash: os.remove(t) 

def calib_img(imglist):

    _imglist = {}
    for im in imglist:
        logger.info(f"processing image {im}")
        f = imglist[im]['filter']
        
        if  os.path.exists(f"{args.caldir}master_sky_flat_img_{f}.fits"):

            with  open('_sci.sof','w') as ff:        
                ff.write(f"{imglist[im]['filepath']} SCIENCE_IMG \n")
                ff.write(f"{args.caldir}master_bias.fits MASTER_BIAS \n")
                ff.write(f"{args.caldir}master_sky_flat_img_{f}.fits MASTER_SKY_FLAT_IMG  \n")
      
            pid = subprocess.Popen(['esorex','fors_img_science','_sci.sof'],
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            output,error = pid.communicate()
            rimg = 'science_reduced_img.fits'
            
        else:
            logger.warning(f"flat field not found for {f}. Correction skipped")
            rimg = imglist[im]['filepath']
     
        pid = subprocess.Popen(['fitscopy',f"{rimg}[0]{trim[args.chip]}",
               f"r_{im}.fits"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        output,error = pid.communicate()

        if not args.nocosmics: clean_cosmics(f"r_{im}")   
        _imglist[im] = imglist[im]
    dfile("object_table_sci_img.fits,phot_background_sci_img.fits,science_reduced_img.fits,_sci.sof,qc*.paf,source_sci_img.fits,esorex.log")

    logger.info("COMPLETED")
    return imglist

def clean_cosmics(img):   #############   correct cosmic rays ##########

   hdulist = fits.open(img+'.fits')
   pixels = hdulist[0].data
   detcosm,mask =  cosmicray_lacosmic(pixels, niter=1,
       sigclip=10.0,  sigfrac=0.3, objlim = 5.0,satlevel=saturat,gain=epadu,
                                      readnoise=ron,verbose=False)
   logger.info(f"{mask.sum()} cosmic pixels flagged ")
   detcosm /= epadu
   imask = np.zeros(detcosm.shape) 
   imask[mask] = 1
   if os.path.exists(img+'.fits'): os.remove(img+'.fits')
   fits.writeto(img+'.fits',detcosm.value,hdulist[0].header)
   if os.path.exists(f'_{img}.mask.fits'): os.remove(f'_{img}.mask.fits')
   fits.writeto(f'_{img}.mask.fits',imask,hdulist[0].header)


def cdither(imglist,group):

    for f in group:
        for mjd in group[f]:
            xshift,yshift = [0.],[0.]
            shifted = []
            for i,o in enumerate(group[f][mjd]):
                data = fits.getdata(f"r_{o}.fits")
                imglist[o]['data'] = data
                mean, median, std = sigma_clipped_stats(data, sigma=3.0)
                imglist[o]['data'] -= median
                daofind = DAOStarFinder(fwhm=4.0, threshold=10.*std)  
                sources = daofind(imglist[o]['data'])  
                ii = np.argsort(sources['flux'])
                sources = sources[ii][:50]
                
                if i == 0:
                    rsources = sources
                    dd = imglist[o]['date'][:10]
                    logger.info(f"{o} reference")
                    continue

                dx,dy,dist = [],[],[]
                step = 3.
                for r in rsources:
                    _dx = r['xcentroid']-sources['xcentroid']
                    _dy = r['ycentroid']-sources['ycentroid']
                    _dist = np.sqrt(_dx**2 + _dy**2)
                    dx.append(_dx)
                    dy.append(_dy)
                    dist.append(_dist)

                dx = np.array(dx)
                dy = np.array(dy)
                dist = np.array(dist)
        
                hist, bins = np.histogram(dist.ravel(),bins=np.arange(0,300.,step))
                dmean = bins[np.argmax(hist)] + step/2.
                ii = np.abs(dist - dmean) < step/2.
                _xshift = np.median(dx[ii])
                _yshift = np.median(dy[ii])
                logger.info(f"{o} {_xshift:6.2f} {_yshift:6.2f}")
                xshift.append(_xshift)
                yshift.append(_yshift)

        shifted = []
        viewer = imexam.connect('ds9')
        for i,o in enumerate(group[f][mjd]):            
            ccd = CCDData(imglist[o]['data'],unit=u.adu)
            if i==0: _shifted = ccd
            else:
                _shifted = transform_image(ccd,shift, shift=(yshift[i],
                                                         xshift[i]))
            shifted.append(_shifted)

        stacked_image = combine(shifted,clip_extrema=True)

        x1 = int(np.max(xshift))+1
        y1 = int(np.max(yshift))+1
        x2 = int(np.min(xshift))
        y2 = int(np.min(yshift))

        ndim = stacked_image.shape
        trim_img = stacked_image[y1:ndim[0]+y2,x1:ndim[1]+x2]
        dfile(f'{f}_{dd}.fits')
        hdr = fits.getheader(f"r_{group[f][mjd][0]}.fits")
        hdr['CRPIX1'] = hdr['CRPIX1']-x1
        hdr['CRPIX2'] = hdr['CRPIX2']-y1
        fits.writeto(f"{f}_{dd}.fits",trim_img,hdr)
        logger.info(f"produced file {f}_{dd}.fits")
         
if __name__ == "__main__":
    
    imglist,group = group_data(args.list)
    if args.list: sys.exit()
    imglist = calib_img(imglist)
    
    cdither(imglist,group)
