# COVID-19 Chest X-Ray Classifier

A convolutional neural network for classifying chest x-rays of patients with COVID-19, viral pneumonia, and normal conditions

This program is based on [the paper](https://www.medrxiv.org/content/10.1101/2020.05.01.20088211v2.full.pdf) by researchers Sohaib Asif, Yi Wenhui, Hou Jin, Yi Tao, and Si Jinhai.

## Data Collection
Data was gathered from three different datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

Additionally, images were augmented with rotations, blur, noise, and flipping using the `augment_images.py` script.

## Models
Two CNN models were generated in `covid_xrays.py`:
* One using a custom series of convolutions and pooling (created with the `generate_model` function)
    * This model performed below the accuracy of the model discussed in the paper referenced above (roughly 94% accuracy in the test dataset).
* Another using transfer learning with the Xception image classifier (created with the `generate_model_transfer` function)
    * This model performed at roughly 97.5% accuracy in the test dataset, near to the researchers' reported accuracy of 98%.
