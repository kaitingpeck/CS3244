import itertools
import imageio
import pandas
import numpy as np
from sklearn import svm
    
from sklearn.model_selection import cross_val_score
from hpsklearn import HyperoptEstimator, svc
import scipy.io
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn import preprocessing
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# Bayesian Optimization
from bayes_opt import BayesianOptimization
from skopt import gp_minimize
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

# File paths
img_src_dir = 'C:/Users/Kai/Desktop/CS3244/Project/data/dataset-resized-128-96/'
label_filepath = 'C:/Users/Kai/Desktop/CS3244/Project/data/labels/zero-indexed-files.txt'
matrix_filepath = 'C:/Users/Kai/Desktop/CS3244/Project/MATLAB-files'
# Training parameters
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
NUM_CLASSES = len(CLASSES)
INPUT_WIDTH = 128
INPUT_HEIGHT = 96
INPUT_DEPTH = 3 #RGB

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def normalize(arr):
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()

        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval))
    return arr

def read_data(label_filepath, img_src_dir):
    '''
    transforms data into the following:
    images = each row is a vector of the pixels of the RGB image
    labels = a column vector representing the respective labels of the image vector
    
    '''
    df = pandas.read_csv(label_filepath, sep = " ", header = None, names = ['images','label'])
    image_names = df['images'].values # list of the image names
    labels = df['label'].values

    vector_length = INPUT_WIDTH * INPUT_HEIGHT * INPUT_DEPTH
    num_images = len(image_names)
    images = np.zeros((num_images, vector_length))
    
    # replace the image name with the respective image vectors
    for i in range(num_images):
        image_name = image_names[i]
        image_vector = imageio.imread(str(img_src_dir + '/' + image_name))
        image_vector = image_vector.flatten() # flatten into a 1-D vector
        images[i] = image_vector # stack the image vectors

    print('Finished reading data. ' + str(num_images) + ' images found in ' + \
          str(NUM_CLASSES) + ' classes.')
    return images, labels

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    title = title + '.png'
    plt.savefig(title, format='png')

images, labels = read_data(label_filepath, img_src_dir)

def random_forest(num_trees, min_leaf_size):
    rf = RandomForestClassifier(n_estimators = int(num_trees), min_samples_leaf = int(min_leaf_size),
                                 random_state=0, oob_score = True, max_features=11)
    score = cross_val_score(rf, images, labels, cv=5).mean()
    return score

def rf_bo():
    gp_params = {"alpha": 1e-5}

    rfBO = BayesianOptimization(random_forest,
    {'num_trees': (10, 5000), 'min_leaf_size': (1, 100)} )
    rfBO.maximize(n_iter=15, **gp_params, acq='ei')
    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % rfBO.res['max']['max_val'])
    print('Params: ' + str(rfBO.res['max']))

rf_bo()
