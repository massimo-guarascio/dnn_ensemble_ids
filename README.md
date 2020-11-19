# dnn_ensemble_ids
DNN-Ensemble IDS is a machine learning based classification model for intrusion detection exploiting ensembles of classifiers. Multiple base models are trained on data gathered in different time windows where different types of attacks occur (Data Chunks). These base classifiers take the form of Deep Neural Networks (DNNs) sharing all the same architecture, but trained against different samples of the given training data. Finally, an incremental learning schema is adopted to cope with different problems such as Large high-speed datastream and rare attacks.

## Author

The code is developed and maintained by Massimo Guarascio and Gianluigi Folino (massimo.guarascio@icar.cnr.it , gianluigi.folino@icar.cnr.it)

## Usage

First, download this repo:
- You need to have 'python3' installed.
- You also need to install 'numpy', 'pandas==1.0.3', and 'sklearn <=0.21', 'imbalanced-learn==0.5.0', 'Keras==2.2.4' and 'tensorflow==1.14.0'.

Then, you can run:

python <seed> <file_output>  [<paramter_file>]

if the parameter file is not specified, the file named 'default.ini' in the current directory is used.
