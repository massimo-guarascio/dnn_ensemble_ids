# dnn_ensemble_ids
DNN-Ensemble IDS is a machine learning based classification model for intrusion detection exploiting a Specialized Ensemble of classifiers. Multiple base classifiers are trained on data gathered in different time windows where different types of attacks occur (Data Chunks). These base classifiers take the form of Deep Neural Networks (DNNs) sharing all the same architecture, but trained against different samples of the given training data. Finally, an incremental learning schema is adopted to cope with different problems such as Large high-speed datastream and rare attacks.


