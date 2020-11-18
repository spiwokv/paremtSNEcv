|lgtm| |lgtmpy|

==========
parmtSNEcv
==========

-------------------------------------------------------
Discovery of collective variables using parametric tSNE
-------------------------------------------------------

Nonlinear dimensionality reduction techniques, such as t-distributed stochastic neighbor embedding (t-SNE) [1]_,
are promising approaches for development of collective variables for enhanced sampling molecular simulations.
Low dimensional embeddings calculated by these methods may be used to define bias forces or bias potentials
in methods such as steered dynamics or metadynamics. The obstacle that must be addressed to apply such collective
variables is caused by the fact that it is usually not possible to calculate low dimensional embeddings for new
out-of-sample structures. To solve this we used so called parametric t-SNE [2]_. Standard t-SNE optimizes
low-dimensional embeddings to match high-dimensional data. Parametric t-SNE uses a feed-forward neuronal
network to calculate low-dimensional embeddings from high-dimensional data. Instead, of optimizing
low-dimensional embeddings it optimizes parameters of the neuronal network. After training, the network can be
used to estimate t-SNE embeddings for new out-of-sample structures.

-------
Install
-------

::

  # comming soon
  pip install parmtSNEcv
  # before
  git clone https://github.com/spiwokv/parmtSNEcv.git
  cd parmtSNEcv
  pip install .

-----
Usage
-----

::

  usage: parmtSNEcv [-h] [-i INFILE] [-p INTOP] [-dim EMBED_DIM] [-perplex PERPLEX]
                    [-boxx BOXX] [-boxy BOXY] [-boxz BOXZ] [-nofit NOFIT]
                    [-layers LAYERS] [-layer1 LAYER1] [-layer2 LAYER2]
                    [-layer3 LAYER3] [-actfun1 ACTFUN1] [-actfun2 ACTFUN2]
                    [-actfun3 ACTFUN3] [-optim OPTIM] [-epochs EPOCHS]
                    [-shuffle_interval SHUFFLE_INTERVAL] [-batch BATCH_SIZE]
                    [-o OFILE] [-model MODELFILE] [-plumed PLUMEDFILE]
  
  Parametric t-SNE using artificial neural networks for development of
  collective variables of molecular systems, requires numpy, keras and mdtraj
  
  optional arguments:
    -h, --help            show this help message and exit
    -i INFILE             Input trajectory in pdb, xtc, trr, dcd, netcdf or
                          mdcrd, WARNING: the trajectory must be 1. centered in
                          the PBC box, 2. fitted to a reference structure and 3.
                          must contain only atoms to be analysed!
    -p INTOP              Input topology in pdb, WARNING: the structure must be
                          1. centered in the PBC box and 2. must contain only
                          atoms to be analysed!
    -perplex PERPLEX      Value of t-SNE perplexity (default 30.0)
    -dim EMBED_DIM        Number of output dimensions (default 2)
    -boxx BOXX            Size of x coordinate of PBC box (from 0 to set value
                          in nm)
    -boxy BOXY            Size of y coordinate of PBC box (from 0 to set value
                          in nm)
    -boxz BOXZ            Size of z coordinate of PBC box (from 0 to set value
                          in nm)
    -nofit NOFIT          Disable fitting, the trajectory must be properly fited
                          (default False)
    -layers LAYERS        Number of hidden layers (allowed values 1-3, default =
                          1)
    -layer1 LAYER1        Number of neurons in the first encoding layer (default
                          = 256)
    -layer2 LAYER2        Number of neurons in the second encoding layer
                          (default = 256)
    -layer3 LAYER3        Number of neurons in the third encoding layer (default
                          = 256)
    -actfun1 ACTFUN1      Activation function of the first layer (default =
                          sigmoid, for options see keras documentation)
    -actfun2 ACTFUN2      Activation function of the second layer (default =
                          linear, for options see keras documentation)
    -actfun3 ACTFUN3      Activation function of the third layer (default =
                          linear, for options see keras documentation)
    -optim OPTIM          Optimizer (default = adam, for options see keras
                          documentation)
    -epochs EPOCHS        Number of epochs (default = 100, >1000 may be
                          necessary for real life applications)
    -shuffle_interval SHUFFLE_INTERVAL
                          Shuffle interval (default = number of epochs + 1)
    -batch BATCH_SIZE     Batch size (0 = no batches, default = 0)
    -o OFILE              Output file with values of t-SNE embeddings
                          (txt, default = no output)
    -model MODELFILE      Prefix for output model files (experimental, default =
                          no output)
    -plumed PLUMEDFILE    Output file for Plumed (default = plumed.dat)

----------
References
----------

.. [1] van der Maaten, L.J.P., Hinton, G.E. (2008) Visualizing Data Using t-SNE.
   *Journal of Machine Learning Research* **9**, 2579-2605.
   
.. [2] van der Maaten, L.J.P. (2009) Learning a Parametric Embedding by Preserving Local Structure.
   *Proceedings of the Twelth International Conference on Artificial Intelligence and Statistics* **5**, 384-391. 

.. |lgtm| image:: https://img.shields.io/lgtm/alerts/g/spiwokv/parmtSNEcv.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/spiwokv/parmtSNEcv/alerts/
    :alt: LGTM code alerts

.. |lgtmpy| image:: https://img.shields.io/lgtm/grade/python/g/spiwokv/parmtSNEcv.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/spiwokv/parmtSNEcv/context:python
    :alt: LGTM python quality


