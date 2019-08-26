# Authors: Kara Ponder, Alex Malz

from astropy.io import fits
import multiprocessing as mp
import numpy as np
import pandas as pd
import sncosmo
import timeit

#start = timeit.timeit()

## this is mostly an SNCOSMO code that KAP updated to be able to return spectra
from sncosmo_spectrum import Spectrum as sncosmo_Spec

def photometry(wavelength, spectrum, variance, band, magnitudes=True, magsystem='ab', **kwargs):
    '''
        band: Check sncosmo for list of bands
        magsystem: Check sncosmo for list of magsystems
        
        bandpasses build into SNCosmo:
        Bessel, SNLS, DES, SDSS, HST ACS WFC, WFC3 IR,
        WFC2 UVIS, Kepler, CSP, JWST NIRCAM/MIRI (nah),
        LSST, keplercam, 4shooter
        '''
    # Define a spectrum object
    #spectrum = sncosmo.Spectrum(wavelength, spectrum)
    spectrum = sncosmo_Spec(wavelength, spectrum, error=variance)
    
    # Calculate flux and flux error in specific band
    flux, fluxerr = spectrum.bandflux(band)
    
    # MagSystem
    mag = sncosmo.get_magsystem(magsystem)
    
    # Calculate magnitudes and mag errors from flux
    magn = mag.band_flux_to_mag(flux, band)
    magerr = 2.5/np.log(10) / flux * fluxerr
    
    if magnitudes:
        return magn, magerr
    else:
        return flux , fluxerr

# Open the SpecObj table with all the Spec info
with fits.open('SpecObj.fits') as hdu_main:
    hdu_main.info()
    df = pd.DataFrame(np.array(hdu_main[1].data).byteswap().newbyteorder())
    df.index = df['CATAID']

header_keyword_dict = {'GAMA': 'CD1_1', 'SDSS': 'CDELT1', 'VVDS': 'CDELT1'}

bandpasses = ['lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty',
              'sdssr', 'sdssi', 'sdssz']

# This was done to deal with the separation of the spectra from a harddrive.
galaxy_data = np.array(['galaxy_data', 'galaxy_data2', 'galaxy_data3',
                        'galaxy_data4', 'galaxy_data5', 'galaxy_data6',
                        'galaxy_data7'])


def helper(n):
    '''
        A helper function to run multi-processing.
        This function goes through every galaxy in the catalog and calculates
        the photometry on the list of bandpasses defined above.
        It only uses spectra with a well defined redshift and for GAMA, SDSS, and VVDS
        as those are the only ones flux calibrated.
    '''
    row = df.loc[df['CATAID'] == n]
    res = [n]
    if row['NQ'].values > 2:
        s = row['SURVEY'].values[0].decode('utf-8').strip()
        if s in ['GAMA', 'SDSS', 'VVDS']:
            f = row['FILENAME'].values[0].decode('utf-8').strip()
            for gd in galaxy_data:
                try:
                    hdu = fits.open('/media/CRP6/Cosmology/GAMA/' + gd + '/' + f)
                except:
                    continue
            spectrum = hdu[0].data[0]
            error = hdu[0].data[1]
            header_info = hdu[0].header
            
            x = np.arange(0, len(spectrum))
            if s in 'SDSS':
                wv_log = header_info['CRVAL1'] + header_info[header_keyword_dict[s]] * x
                wv = 10**(wv_log)
            else:
                wv = header_info['WMIN'] + header_info[header_keyword_dict[s]] * x

            wv = wv * (1. + row['Z'].values[0])

            if s in 'VVDS':
                spectrum = spectrum/1e-17
                error = error/1e-17

            # Now I need to clean out the nans from the flux.
            wh_no_err, = np.where(np.isinf(error) | np.isnan(error))
            wh_no_flux, = np.where(np.isinf(spectrum) | np.isnan(spectrum))

            error[wh_no_err] = 0.0
            spectrum[wh_no_err] = 0.0

            error[wh_no_flux] = 0.0
            spectrum[wh_no_flux] = 0.0

            for i, b in enumerate(bandpasses):
                try:
                    mag, mag_err = photometry(wv, spectrum*1e-17, error*1e-17, b)
                    res.append(mag)
                    res.append(mag_err)
                except:
                    mag, mag_err = 0., 0.
                    res.append(mag)
                    res.append(mag_err)
                    continue

            hdu.close()
    return(res)


## This is the start of the actual calculations.
nps = mp.cpu_count()
pool = mp.Pool(nps - 1)
phot_measures = pool.map(helper, df['CATAID'])

output = pd.DataFrame(phot_measures)

# Save the output to the CSV file. Later we add to SpecObj
# This one was used in production, but putting the right way to save below.
# output.to_csv('photo.csv')

output["CATAID"] = pd.to_numeric(output["CATAID"])
output.index = output['CATAID']

new = df.merge(output, on='CATAID')
new.to_csv('SpecObjPhot.csv')

