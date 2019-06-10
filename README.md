# pattern-image-analysis
![Example crystal growth](https://github.com/jsbangsund/pattern-image-analysis/blob/master/example%20images/feature_image.png)

This repository includes example Python code to extract periodic pattern wavelength from microscopy images using Fast Fourier Transforms (FFT) and to assess pattern quality based on the magnitude of the low frequency portion of the FFT. This code was developed to analyze the images in our paper on spontaneous pattern formation in organic semiconductors:

Formation of aligned periodic patterns during the crystallization of organic semiconductor thin films. Nature Materials 1 (2019). doi: [10.1038/s41563-019-0379-3](https://doi.org/10.1038/s41563-019-0379-3).

Much more sophisticated packages exist for this sort of analysis (e.g. [PyFAI](https://pyfai.readthedocs.io/en/latest/)), but these notebooks are shared to provide an example for how to analyze large image sets with relative ease.

Limited example images are included to illustrate the analysis functions. Some of the code cells in the provided Jupyter Notebooks will not run due to unavailable images. If you would like the full image sets, please contact me at jsb@umn.edu.
