# -*- coding: utf-8 -*-
"""

@author: Lucas Drumetz

This script applies the ELMM algorithm and the RELMM algorithm on a subset of 
the 2013 DFC Houston hyperspectral dataset, acquired above the University of 
Houston Campus, in June 2012. We use a small part of the hyperspectral dataset 
provided courtesy of the Hyperspectral Image Analysis group and the NSF Funded
Center for Airborne Laser Mapping at the University of Houston, and used
in the 2013 Data Fusion Contest (DFC):

C. Debes et al., "Hyperspectral and LiDAR Data Fusion: Outcome of the
2013 GRSS Data Fusion Contest," in IEEE Journal of Selected Topics in
Applied Earth Observations and Remote Sensing, vol. 7, no. 6,
pp. 2405-2418, June 2014.

Details for the ELMM and the associated algorithm can be found here:

L. Drumetz, M. Veganzones, S. Henrot, R. Phlypo, J. Chanussot and C. Jutten, 
"Blind Hyperspectral Unmixing Using an Extended Linear Mixing Model to Address
Spectral Variability," in IEEE Transactions on Image Processing, vol. 25, 
no. 8, pp. 3890-3905, Aug. 2016.
doi: 10.1109/TIP.2016.2579259

Note that the version implemented here does not include the spatial 
regularization on the endmembers and abundances.

Details for the RELMM algorithm can be found here:

L. Drumetz, J. Chanussot, C. Jutten, W. Ma and A. Iwasaki, "Spectral 
Variability Aware Blind Hyperspectral Image Unmixing Based on Convex Geometry,"
in IEEE Transactions on Image Processing, vol. 29, pp. 4568-4582, 2020.
doi: 10.1109/TIP.2020.2974062

Author: Lucas Drumetz
Version: 1.1
Latest Revision: 20th March 2020

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import time
from rescale import rescale
from SCLSU import SCLSU
from pca_viz import pca_viz
from ELMM import ELMM
from RELMM import RELMM


plt.close("all")

data = scipy.io.loadmat("real_data_1.mat")

im = rescale(data['data'])

rgb = im[:,:,[56,29,19]]

uint8_rgb = rescale(rgb)

plt.imshow(uint8_rgb, interpolation='nearest')

endmembers = scipy.io.loadmat("endmembers_houston.mat")

S0 = endmembers['S0']

P = S0.shape[1]

plt.figure(1)
plt.plot(S0)

[m,n,L] = im.shape

N = m*n;

imr = np.transpose(im.reshape((N,L),order='F'))

# unmixing with SCLSU

start = time.clock()
[A_SCLSU,psi_SCLSU,S_SCLSU] = SCLSU(imr,S0)
end = time.clock()
print(end - start)


A_SCLSU_im =(np.transpose(A_SCLSU)).reshape((m,n,P),order = 'F')


plt.figure(2)

plt.subplot(341)
plt.imshow(A_SCLSU_im[:,:,0])
plt.ylabel('SCLSU')
plt.xlabel('Vegetation')
plt.subplot(342)
plt.imshow(A_SCLSU_im[:,:,1])
plt.xlabel('Red Roofs')
plt.subplot(343)
plt.imshow(A_SCLSU_im[:,:,2])
plt.xlabel('Concrete')
plt.subplot(344)
plt.imshow(A_SCLSU_im[:,:,3])
plt.xlabel('Asphalt')

    
plt.figure(3)
plt.subplot(341)
plt.imshow(psi_SCLSU.reshape(m,n,order = 'F')) 
plt.ylabel('SCLSU')
plt.xlabel('Vegetation')
plt.colorbar()
plt.subplot(342)
plt.imshow(psi_SCLSU.reshape(m,n,order = 'F')) 
plt.xlabel('Red Roofs')
plt.colorbar()
plt.subplot(343)
plt.imshow(psi_SCLSU.reshape(m,n,order = 'F')) 
plt.xlabel('Concrete')
plt.colorbar()
plt.subplot(344)
plt.imshow(psi_SCLSU.reshape(m,n,order = 'F'))  
plt.xlabel('Asphalt') 
plt.colorbar()

# scatterplot visualization

pca_viz(imr,S_SCLSU.reshape((L,P*N),order = 'F'))

#  ELMM 
  
lambda_S = 0.5  
  

[A,psi,S] = ELMM(imr,A_SCLSU,S0,lambda_S)
  
A_im =(np.transpose(A)).reshape((m,n,P),order = 'F')
psi_im = (np.transpose(psi)).reshape((m,n,P),order = 'F')

plt.figure(2)

plt.subplot(345)
plt.imshow(A_im[:,:,0])
plt.ylabel('ELMM')
plt.xlabel('Vegetation')
plt.subplot(346)
plt.imshow(A_im[:,:,1])
plt.xlabel('Red Roofs')
plt.subplot(347)
plt.imshow(A_im[:,:,2])
plt.xlabel('Concrete')
plt.subplot(348)
plt.imshow(A_im[:,:,3])
plt.xlabel('Asphalt')
plt.figure()

plt.figure(3)
plt.subplot(345)
plt.imshow(psi_im[:,:,0])
plt.ylabel('ELMM')
plt.xlabel('Vegetation')
plt.colorbar()
plt.subplot(346)
plt.imshow(psi_im[:,:,1])
plt.xlabel('Red Roofs')
plt.colorbar()
plt.subplot(347)
plt.imshow(psi_im[:,:,2])
plt.xlabel('Concrete')
plt.colorbar()
plt.subplot(348)
plt.imshow(psi_im[:,:,3])
plt.xlabel('Asphalt')
plt.colorbar()

  
pca_viz(imr,S.reshape((L,P*N),order = 'F'))

# RELMM (this section of the code takes much longer to run than the other ones)

lambda_S = 2  
lambda_S0 = 5 
  

S0_norm = np.zeros(S0.shape)  
  
for p in np.arange(P):
    S0_norm[:,p] = S0[:,p]/np.linalg.norm(S0[:,p])
  
plt.figure()
plt.plot(S0_norm)
 
[A_RELMM,psi_RELMM,S_RELMM,S0_final_RELMM] = RELMM(imr,A_SCLSU,S0_norm,lambda_S,lambda_S0)

A_RELMM_im =(np.transpose(A_RELMM)).reshape((m,n,P),order = 'F')
psi_RELMM_im = (np.transpose(psi_RELMM)).reshape((m,n,P),order = 'F')
  
pca_viz(imr,S_RELMM.reshape((L,P*N),order = 'F'))  


plt.figure(2)
plt.subplot(349)
plt.imshow(A_RELMM_im[:,:,0])
plt.ylabel('RELMM')
plt.xlabel('Vegetation')
plt.subplot(3,4,10)
plt.imshow(A_RELMM_im[:,:,1])
plt.xlabel('Red Roofs')
plt.subplot(3,4,11)
plt.imshow(A_RELMM_im[:,:,2])
plt.xlabel('Concrete')
plt.subplot(3,4,12)
plt.imshow(A_RELMM_im[:,:,3])
plt.xlabel('Asphalt')
plt.figure()

plt.figure(3)
plt.subplot(349)
plt.imshow(psi_RELMM_im[:,:,0])
plt.ylabel('RELMM')
plt.xlabel('Vegetation')
plt.colorbar()
plt.subplot(3,4,10)
plt.imshow(psi_RELMM_im[:,:,1])
plt.xlabel('Red Roofs')
plt.colorbar()
plt.subplot(3,4,11)
plt.imshow(psi_RELMM_im[:,:,2])
plt.xlabel('Concrete')
plt.colorbar()
plt.subplot(3,4,12)
plt.imshow(psi_RELMM_im[:,:,3])
plt.xlabel('Asphalt')
plt.colorbar()
