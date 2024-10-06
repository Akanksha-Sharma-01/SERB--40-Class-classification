# SERB-40-Class-classification
This repository contains codes and models for 40 class classification of EEG signal against visual stimuli for SERB. The two approaches used for the task are: a) CNN-BiLSTM b) CNN-Transformer. 
EEG-ImageNet is the dataset used which contains data of 6 subjects for 50 images of each class. This dataset contains some non-uniformities which are handled carefully by keeping data of 49 images for each class and removing one class as a whole with irregular data among subjects. The resultant data contains 39 classes which are uniformly distributed. The uniformity of data is also maintained while splitting data into 80% training, 10% validation and 10% test data. Each split contains data of all 6 subjects for all classes. Further, data corresponding to single image for all subjects is taken into single split i.e. data of all subjects for one image is either in train or in validation or in test.
The code and model is present in the repository. In order to run the model to get the desired result, the system should have following requirements:
tensorflow (version 2.1.0)
python (version 3.10.13)
scikit-learn (version 1.5.2)
numpy (version 1.26.3)
