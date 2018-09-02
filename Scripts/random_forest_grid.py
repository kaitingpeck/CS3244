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
from matplotlib import gridspec

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

    # scipy.io.savemat(matrix_filepath + '/images-' + str(INPUT_WIDTH) + '.mat', dict(images=images))
    # scipy.io.savemat(matrix_filepath + '/labels.mat', dict(labels=labels))
    
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
    title = title + '.jpg'
    plt.savefig(title, format='jpeg')

'''def random_forest(X_train, y_train, num_trees):
    rf = RandomForestClassifier(n_estimators = num_trees, random_state=0, oob_score = True, criterion='gini')
    rf.fit(X_train, y_train)
    return rf'''
  
def k_fold_rf(images, labels, num_trees, num_folds=5):
    kf = StratifiedKFold(n_splits=5)
    i = 1
    for train_index, val_index in kf.split(images, labels):
        X_train, X_val = images[train_index], images[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        # Obtained trained model using this set of training data
        model = random_forest(X_train, y_train, num_trees)
        print('Model training completed.')

        # save the model to disk
        #filename = 'finalized_model-' + str(num_trees) + '-trees-' + str(i) + '-fold' + '.sav'
        #pickle.dump(model, open(filename, 'wb'))
        #print('Model saved.')
        
        # Report ooberror
        print(str(num_trees) + '-trees-' + str(i) + '-th fold, Out-of-bag (accuracy) estimate: ' + str(model.oob_score_))

        # Run model on validation data
        y_pred = model.predict(X_val)
       
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_val, y_pred)
        np.set_printoptions(precision=2)

        if i==1:
            # Plot normalized confusion matrix
            plt.figure()
            plot_title = 'Normalized Confusion Matrix'
        
            plot_confusion_matrix(cnf_matrix, classes=CLASSES, normalize=True,
                                  title= plot_title)
        
        # set index for next fold
        i += 1

def random_forest(X_train, y_train, num_trees, num_folds=5):
    rf = RandomForestClassifier(n_estimators = int(num_trees), random_state=0, oob_score = True, criterion='gini')
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()


def run_rf_diff_trees(X, y, num_folds=5):
    x = np.linspace(100, 1000, 19).reshape(-1,1)
    print(x)
    param_trees = x.reshape(1,19).tolist()[0]
    scores = []
    for num_trees in param_trees:
        print('... Running random forest with ' + str(num_trees) + ' trees ...')
        score = random_forest(X, y, num_trees, 5)
        print(score)
        scores.append(score)
    highest = max(scores)
    idx = scores.index(highest)
    print('Highest accuracy: ' + str(highest))
    print('Achieved with index: ' + str(param_trees[idx]) + ' trees')
        
    fit = np.polyfit(param_trees,scores,2)
    fit_fn = np.poly1d(fit) 
    # fit_fn is now a function which takes in x and returns an estimate for y

    
    #plt.xlim(0, 5)
    #plt.ylim(0, 12)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Scores vs no. of trees in random forest', fontdict={'size':30})
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    axis.plot(param_trees,scores, 'bo', param_trees, fit_fn(param_trees), '--k')
    # axis.plot(param_trees, scores, 'b+')
    axis.set_ylabel('Score', fontdict={'size':20})
    axis.set_xlabel('No. of trees', fontdict={'size':20})
    fig.show()

images, labels = read_data(label_filepath, img_src_dir)
run_rf_diff_trees(images, labels, 5)
