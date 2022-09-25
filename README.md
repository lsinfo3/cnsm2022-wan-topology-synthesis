# Comparing Traditional and GAN-based Approaches for the Synthesis of Wide Area Network Topologies

This repository contains the source code and (generated) datasets used in the CNSM2022 paper *Comparing Traditional and GAN-based Approaches for the Synthesis of Wide Area Network Topologies*. Below is an overview of the contained folders and files.

### *adj_matrices*

### *plot_data* and *data*

### \<network\>\_naive.zip

These archives contain the generated networks for all of the investigated generators.

For the *traditional* generations (2K, ER, BA, WS), it contains 1000 files each in the following format:

*synth\_sample_\<generator\>\_ i.pkl*, which are NetworkX graphs (*nx.Graph()*).

For the GAN, the contained data is more complex. First, there are files named *image\_at\_epoch\_1000_s_(BW|RGB).pdf*, which showcase **samples** of the generated networks for each of the then seeds *s*, just for some visualization, other than that, they have **no** further use.

Furthermore, for the GAN, there are files named *sample\_at\_epoch\_1000_s\_i_(BW|RGB).pkl*, which are the raw files that the GAN produced with seed *s*. These are NumPy Arrays (*np.array()*) with dimensions *n x n x 1* for BW and *n x n x 3* for RGB.

Next and last, there are files named *synth_sample\_i\_s\_weightssampled?\_(BW|RGB).pkl*. These are the postprocessed samples for the GAN, for seed *s* and *weightssampled?* is a boolean (so, 0 or 1), which specifies if the weights were sampled onto the graph. Like for the traditional generators, these are NetworkX graphs (*nx.Graph()*).

### \<network\>\_hierarchical.zip

### eval\_(traditional|gan)\_(naive|hierarchical).py

### traditional.py, wangan.py, WANGAN-simple.py and WANGAN-hierarchical.py

### read_zoo.py

