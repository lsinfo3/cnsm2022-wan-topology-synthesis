# Comparing Traditional and GAN-based Approaches for the Synthesis of Wide Area Network Topologies

This repository contains the source code and (generated) datasets used in the CNSM2022 paper *Comparing Traditional and GAN-based Approaches for the Synthesis of Wide Area Network Topologies*. Below is an overview of the contained folders and files. [More info below to come :)]

### *adj_matrices*

Adjancecy matrices of the networks extracted from the OOS. See *code* for more info.

### *plot_data* and *plots*

As the name suggests, the plots and corresponding that from the paper, for all generators (2K, BA, WS, ER, for GAN: RGB and BW). For the hierarchical case the endings "_2", "_3", "_4" illustrate the number of clusters. Note that the data for the "weighted DC" as defined in the paper can be derived from the "Weights" column (which contains the sum of all edge weights of the whole network) by computing **Weights*2/Nodes**, since the average weighted DC is simply just two times the sum of all network weights (since each edge is counted in twice) divided by the number of nodes (it's what is shown in the plots, just for clarification).

### *synthetic\_networks*

Contains archives that contain the generated networks for all of the investigated generators for the four real topologies, including one hierarchical use case.

For the *traditional* generations (2K, ER, BA, WS), it contains 1000 files each in the following format for the naive approach:

*synth\_sample_\<generator\>\_ i.pkl*, which are NetworkX graphs (*nx.Graph()*).

For the GAN, the contained data is more complex. First, there are files named *image\_at\_epoch\_1000_s_(BW|RGB).pdf*, which showcase **samples** of the generated networks for each of the then seeds *s*, just for some visualization, other than that, they have **no** further use.

Furthermore, for the GAN, there are files named *sample\_at\_epoch\_1000_s\_i_(BW|RGB).pkl*, which are the raw files that the GAN produced with seed *s*. These are NumPy Arrays (*np.array()*) with dimensions *n x n x 1* for BW and *n x n x 3* for RGB.

Next and last, there are files named *synth_sample\_i\_s\_weightssampled?\_(BW|RGB).pkl*. These are the postprocessed samples for the GAN, for seed *s* and *weightssampled?* is a boolean (so, 0 or 1), which specifies if the weights were sampled onto the graph. Like for the traditional generators, these are NetworkX graphs (*nx.Graph()*).

For the single hierarchical example, it contains multiple subfolders for local and global views.

### *code*

Contains the source code, including some helper functions for transforming the .ned files from the OOS into the matrices of the adj_matrices folder.

*wangan.py* is the main functionality that trains the GANs and saves in the end the generated samples.

*WANGAN-simple.py* and *WANGAN-hierarchical* call the previous script to either perform the naive approach or the hierarchical approach. 

<sup>Note that since we trained the GANs on GPU, results may differ even though seeds are set (e.g., due to parallelization or non-deterministic GPU ops), especially for the RGB-based GANs; but general trends should remain the same. For TensorFlow >= 2.9.0 there are now options via *tf.config.experimental.enable_op_determinism*, so you might want to check this out, since research on our work started prior to that release.</sup>

There are four *eval* scripts, two for the naive approach for GAN-based and traditional approaches, and two for the hierarchical approach. Note that the traditionally generated network are created in an ad-hoc manner by calling the *traditional.py* script due to their simplicity here (can be controlled via *save* and load *parameters*, if this is not wanted).

Lastly, there is the *read_zoo_revised.py* script, which takes the .ned files from the OOS (https://github.com/lsinfo3/OpenFlowOMNeTSuite/tree/master/openflow/networks) and transforms them into the adjacency matrices, weighted and unweighted ones.
