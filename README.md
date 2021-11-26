# Landmark Classification using CNNs

Implementation of the Landmark Classification Project in the Udacity Deep Learning Nanodegree.

## Requirements

Landmarks Dataset:

https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip

### Step 1: Create a CNN to Classify Landmarks (from Scratch)

* The submission includes the required notebook file and HTML file. When the HTML file is created, all the code cells in the notebook need to have been run so that reviewers can see the final implementation and output.
* The submission randomly splits the images at landmark_images/train into train and validation sets. The submission then creates a data loader for the created train set, a data loader for the created validation set, and a data loader for the images at landmark_images/test.
* Answer describes each step of the image preprocessing and augmentation. Augmentation (cropping, rotating, etc.) is not a requirement.
* The submission displays at least 5 images from the train data loader, and labels each image with its class name (e.g., "Golden Gate Bridge").
* The submission chooses appropriate loss and optimization functions for this classification task.
* The submission specifies a CNN architecture.
* Answer describes the reasoning behind the selection of layer types.
* The submission implements an algorithm to train a model for a number of epochs and save the "best" result.
* The submission implements a custom weight initialization function that modifies all the weights of the model. The submission does not cause the training loss or validation loss to explode to nan.
* The trained model attains at least 20% accuracy on the test set.

### Step 2: Create a CNN to Classify Landmarks (using Transfer Learning)

* The submission specifies a model architecture that uses part of a pre-trained model.
* The submission details why the chosen architecture is suitable for this classification task.
* The submission uses model checkpointing to train the model and saves the model weights with the best validation loss.
* Accuracy on the test set is 60% or greater.

### Step 3: Write Your Landmark Prediction Algorithm

* The submission implements functionality to use the transfer learned CNN from Step 2 to predict top k landmarks. The returned predictions are the names of the landmarks (e.g., "Golden Gate Bridge").
* The submission displays a given image and uses the functionality in "Write Your Algorithm, Part 1" to predict the top 3 landmarks.
* The submission tests at least 4 images.
* Submission provides at least three possible points of improvement for the classification algorithm.

### Optional goals

* Keep iterating on your model and training parameters to see how high you can get your test accuracy! One example idea to experiment with is to apply different augmentations to your training data.
* Use the features from the penultimate layer of your from-scratch CNN or transfer-learned CNN to implement an image retrieval algorithm. Your algorithm should roughly perform the following procedure: given an image, extract the CNN features for the image, compute the dot product between the aforementioned CNN features and the CNN features for each of the images in landmark_images, return the images that have the highest dot product values.
* Include in your submission discussion around additional use cases of your model - what other situations might it be useful?
