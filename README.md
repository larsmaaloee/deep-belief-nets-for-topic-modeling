Deep-Belief-Nets-for-Topic-Modeling
===================================

This repository is a proof of concept toolbox for using Deep Belief Nets for Topic Modeling in Python. The toolbox was implemented while writing a Master Thesis (http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/6742/pdf/imm6742.pdf) on Topic Modeling.

The toolbox is a work in progress and must be considered as a prototype. Please refer to the Master Thesis for a thorough explanation on the implementation and feel free to contribute.


Implementation
==============
The implementation consists of 3 main modules:
  - Data preparation
  		The BOWs are generated in a format understood by the toolbox. The toolbox applies to batch learning.
  - DBN
  		Pretraining and finetuning. 
  - Testing
  		Evaluate the performance of the generated network through benchmarks and visualizations.

How to
======
In 'main.py' is an example on how to run the toolbox on the 20 newsgroups data set (http://qwone.com/~jason/20Newsgroups/). Work from this example to learn how the toolbox works.
