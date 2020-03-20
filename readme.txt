This toolbox contains several scripts and functions in Python, to unmix hyperspectral data using the Extended Linear Mixing Model (ELMM).

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

