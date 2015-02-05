Deep Belief Nets for Topic Modeling
===================================
Python toolbox using deep belief nets (DBN) for running topic modeling on document data. The concept of the method is to load bag-of-words (BOW) and produce a strong latent representation that will then be used for a content based recommender system.

The toolbox is written for a [M.Sc. thesis project](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6742). For a shorter read we urge you to read the article [Deep Belief Nets for Topic Modeling](http://xxx.tau.ac.il/abs/1501.04325) accepted at the ICML2014 workshop Knowledge-Powered Deep Learning for Text Mining (KPDLTM).

The toolbox is tested to run on Windows 7, Ubuntu 14.04.1 and OSX 10.8-10. You need following prerequisite packages: nltk, numpy, scipy, scikit-learn and matplotlib installed on your system before running the toolbox. If you are interested in producing 3D plots of the output space you will need to install MENCODER and FFMPEG (only tested on OSX).

![PCA on the output of 6 categories from the 20 newsgroups dataset run on a 2000-500-250-125-10 (real output) DBN.](/output/_output_20news.jpg?raw=true "20 newsgroups output")

Running the Toolbox
===================
In the [main.py](main.py) python file you will find 3 examples on how to run the toolbox:

Example 1
---------
In order to run this example you will need to download the 20 Newsgroup dataset [20news-bydate.tar.gz](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) from http://qwone.com/~jason/20Newsgroups/ and place the unpacked dir "20-news-bydate" in the "./input" dir.

The execution order is as follows:
- Stem the documents in the training and test set.
- Initialise the data processing module and generate the BOWs for the training and test set.
- Initialise the DBN (shape: 2000-500-500-128 binary output units) and pretrain followed by finetuning for 50 epochs each.
- Evaluate the accuracy of the trained network by performing forward pass of the test set and comparing the nearest neighbors in the output space.
- Visualise the test set on 6 categories using PCA.

Example 2
---------
In order to run this example you will need to download the 20 Newsgroup dataset [20news-18828.tar.gz](http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz) from http://qwone.com/~jason/20Newsgroups/ and place the unpacked dir "20news-18828" in the "./input" dir.

The execution order is as follows:
- Stem the documents in the data set.
- Initialise the data processing module and generate the BOWs for the training (70% of the docs) and test set (30% of the docs).
- Initialise the DBN (shape: 2000-500-250-125-10 real output units) and pretrain followed by finetuning for 50 epochs each.
- Evaluate the accuracy of the trained network by performing forward pass of the test set and comparing the nearest neighbors in the output space.
- Visualise the test set on 6 categories using PCA.

Example 3
---------
In the "./output" dir is a compressed file "_20news-19997.zip". These are the output files after running the DBN (shape: 2000-500-250-125-10 real output units) on the [20news-19997.tar.gz](http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz) from http://qwone.com/~jason/20Newsgroups/ for 50 epochs pretraining and finetuning. Unzip the compressed chunks by running the shell script "output/_unzip.sh".

The execution order is as follows:
- Evaluate the accuracy of the trained network by performing forward pass of the test set and comparing the nearest neighbors in the output space.
- Visualise the test set on 6 categories using PCA.

Running the toolbox on other datasets
-------------------------------------
The toolbox apply to all text datasets as long as the execution order is followed (cf. Examples 1 and 2):
- Stem documents.
- Generate BOWs.
- Initialise the DBN.
- Pretrain.
- Finetune.
- Evaluate v Visualise.

Please note that many of the learning parameters are hardcoded into the pretraining and finetuning modules. The current setting has proven to work on various datasets.

During execution all data is saved to the harddrive which slows down the execution, but will eliminate any out-of-memory errors. Furthermore it gives the analyst the ability to resume the training at a random point in training even with different parameters.

Acknowledgements
----------------
(cf. the article or M.Sc. thesis mentioned in the beginning for proper citations to litterature used in order to realize this toolbox.)
- Geoffrey Hinton and Ruslan Salakhutdinovs work on DBNs for dimensionality reduction, restricted boltzmann machines and replicated softmax models.
- Roland Memisevic Python interpretation of Carl Edward Rasmussens Conjugate Gradient script.

Note from author
----------------
Please do not hessitate to contact or contribute if any errors or ideas occur. Enjoy.

Best regards

Lars Maaloee, 
[PHD student](http://orbit.dtu.dk/en/persons/lars-maaloee(0ba00555-e860-4036-9d7b-01ec1d76f96d).html), 
Technical University of Denmark, 
[LinkedIn](http://dk.linkedin.com/in/larsmaaloe)


