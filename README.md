![Classification_Neural_Network](https://user-images.githubusercontent.com/99198862/205621099-e169a879-023b-44a7-8bd9-8f7e0c8a313e.png)
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Usage](#usage)

## General info
A Neural Network for classifying data sets, contains several advanced optimization parameters. Includes plotting and logging functionality.
	
## Technologies
Project is created with:
* Python 3.10.2
* Numpy 1.23.4
* SciPy 1.9.3
* Matplotlib 3.6.2
	
## Setup
To run this project, first install the dependencies from requirements.txt by running the following command in the terminal

![image](https://user-images.githubusercontent.com/99198862/205624199-58676487-2940-4169-a944-1afe3f0c2a04.png)


## Usage
Create a new file and import ann.py and create a Neural_Network class

![image](https://user-images.githubusercontent.com/99198862/205624081-981237fc-f98a-4bc9-a1de-db794341fd31.png)

Most data sets come with labels and training data in one file, split_data_set() splits them up and also divides the data into training, validation and testing.

![image](https://user-images.githubusercontent.com/99198862/205625990-b496721a-38c5-4b11-a5c9-b5dcb9b6d1a9.png)

Next write down the labels of your data set

![image](https://user-images.githubusercontent.com/99198862/205626776-1c57dcf1-1396-45d5-96af-47e8c42b83ea.png)

Then configure hyper-parameters.

![image](https://user-images.githubusercontent.com/99198862/205626725-de68fcec-ecfd-480c-b3f6-d656d8296f34.png)

The last line is ```start_training()``` which will begin training.

While its running you will get regular updates of how the network is performing,

![image](https://user-images.githubusercontent.com/99198862/205627402-cb6896c1-7d4d-4031-96a8-63078d405254.png)
