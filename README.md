# COVID-19 Chest X-Ray Classifier

A convolutional neural network for classifying chest x-rays of patients with COVID-19, viral pneumonia, and normal conditions

This program is based on [the paper](https://www.medrxiv.org/content/10.1101/2020.05.01.20088211v2.full.pdf) by researchers Sohaib Asif, Yi Wenhui, Hou Jin, Yi Tao, and Si Jinhai. In that paper, the researchers achieved about 98% accuracy on test data when using a CNN model with Inception v3 transfer learning.

## Data Collection
Data was gathered from three different datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
    * So far, only this dataset has been used; the computer executing this program runs out of memory when introducing any additional images. 
    * A memory upgrade is planned for the future to perform more complete model training.
* https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

Additionally, images were turned to grayscale and augmented with rotations, blur, noise, and flipping using the `augment_images.py` script.

## Data Usage
Images used from the above datasets were chest x-rays that depicted either
* COVID-19
* Viral pneumonia (VP)
* Or normal conditions

These images were shuffled and split into training (80%), validation (10%), and testing (10%) datasets, where
* The validation and testing datasets had an equal split of COVID-19, VP, and normal x-rays.
* The training dataset consisted of 10% more VP and normal x-rays than COVID-19 x-rays.
    * This was done to utilize more images from the above datasets (while also attempting to avoid significant bias), since a much smaller proportion of x-rays collected depicted COVID-19 patients. 
    * Training with an even split of x-ray images has not yet been tested.

## Models
Three CNN models were generated in `covid_xrays.py`:
1. Using a custom series of convolutions and pooling (created with the `generate_model` function)
    * Accuracy (with testing data): about 93%
2. Using transfer learning with the Inception v3 image classifier (created with the `generate_model_inception_v3` function)
    * Accuracy (with testing data): about 96%
3. Using transfer learning with the Xception image classifier (created with the `generate_model_xception` function)
    * Accuracy (with testing data): about 97%

## Additional Notes
This program uses the [discord_notify](https://github.com/MatthewATaylor/discord_notify) package to log the program's status through a Discord webhook.
