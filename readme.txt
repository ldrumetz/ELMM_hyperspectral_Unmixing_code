This toolbox contains several scripts and functions in Python, to unmix hyperspectral data using the Extended Linear Mixing Model (ELMM) and some variants

Details about the ELMM can be found here:

L. Drumetz, M. Veganzones, S. Henrot, R. Phlypo, J. Chanussot and C. Jutten, 
"Blind Hyperspectral Unmixing Using an Extended Linear Mixing Model to Address
Spectral Variability," in IEEE Transactions on Image Processing, vol. 25, 
no. 8, pp. 3890-3905, Aug. 2016.
doi: 10.1109/TIP.2016.2579259

And more theoretical motivation for the model can be found here:

L. Drumetz, J. Chanussot, C. Jutten, W. Ma and A. Iwasaki, "Spectral 
Variability Aware Blind Hyperspectral Image Unmixing Based on Convex Geometry,"
in IEEE Transactions on Image Processing, vol. 29, pp. 4568-4582, 2020.
doi: 10.1109/TIP.2020.2974062

Physical motivation of the model can be found here:

L. Drumetz, J. Chanussot and C. Jutten, "Spectral Unmixing: A Derivation of the Extended Linear Mixing Model From the Hapke Model," in IEEE Geoscience and Remote Sensing Letters.
doi: 10.1109/LGRS.2019.2958203

And its ability to account for nonlinear mixing phenomena to some extent can be found here:

L. Drumetz, B. Ehsandoust, J. Chanussot, B. Rivet, M. Babaie-Zadeh and C. Jutten, "Relationships Between Nonlinear and Space-Variant Linear Models in Hyperspectral Image Unmixing," in IEEE Signal Processing Letters, vol. 24, no. 10, pp. 1567-1571, Oct. 2017.
doi: 10.1109/LSP.2017.2747478

The contents include:

- ELMM.py: Function performing the unmixing with the ELMM
- RELMM.py: Function performing the unmixing with the Robust version of the ELMM
- demo_ELMM.py: example of use of the algorithms on a real hyperspectral dataset
- real_data_1.mat: real dataset used (crop of the DFC 2013 data)
- endmembers_houston.mat: reference endmember matrix used in the demo
- FCLSU.py : function performing the standard fully constrained least squared unmixing.
- SCLSU.py: function performing a scaled version of CLSU, which follows a particular case of the ELMM
- pca_viz.py: function projecting data and endmembers on the space spanned by the first three principal components of the data and displaying a scatterplot
- rescale.py: function rescaling hyperspectral data between 0 and 1
- proj_simplex.py: function projecting one or several vectors onto the unit simplex

