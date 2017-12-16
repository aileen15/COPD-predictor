#--------------------------------------------------------------------
# prediction.py:  Predicting County Level Cost Differences for Treating
#                 Chronic Obstructive Pulmonary Disease
# Jonathan Lin, Michael Smith, Aileen Wang
# Stanford University
# jolituba@stanford.edu,  msmith11@stanford.edu,  aileen15@stanford.edu
# December 15, 2017
#---------------------------------------------------------------------
import matplotlib.pyplot as plt
import csv

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os, random, operator, sys
import datetime
from optparse import OptionParser

#----------------------------------------------------------------------------------
# Parsing the input csv file and initialzing the feature dataset for the prediction
#----------------------------------------------------------------------------------
def parseCSVfile(csv_file, binary_classify, year, single_year, num_classes):
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        records = list(reader)
    data = []
    keys = records[0] #use the header of csv file as the keys
    cols = len(keys)
    
    startIndex = 4
    for j in range(startIndex, cols-1):
        # calculate the average postive sum and negative sum
        psum = 0
        pcount = 0
        nsum = 0
        ncount = 0
        for i in range(1, len(records)):           
            val = float(records[i][cols-1])
            if val > 0:
                psum += val
                pcount = pcount + 1
            elif val < 0:
                nsum += val
                ncount = ncount + 1
        if pcount > 0:
            psum /= pcount
        if ncount > 0:
            nsum /= ncount        
   
    for i in range(1, len(records)):
        if year == "" or records[i][0] == year:
            value = float(records[i][cols-1])
            # set the target y value for classification
            if binary_classify == True:
                y = 1 if value > 0 else 0              
            else:
                if single_year == False:
                    if value < -0.20:
                        y = -2
                    elif value < -0.10:
                        y = -1
                    elif value < 0:
                        y = 0
                    elif value < 0.10:
                        y = 1
                    elif value < 0.20:
                        y = 2
                    else:
                        y = 3
                else:                   
                    if num_classes == 3:
                        if value < nsum:
                            y = -1                
                        elif value < psum:
                            y = 0
                        else:
                            y = 1
                    else:
                        if value < nsum:
                            y = -2
                        elif value < 0:
                            y = -1
                        elif value < psum:
                            y = 1
                        else:
                            y = 2
            features = {}
            for j in range(startIndex, cols-1):                
                features[keys[j]] = float(records[i][j])
                    
            data.append((features, y))        
    print('Number of rows =' + str(len(data)))
    return data

#-----------------------------------------------------------------
# Draw the comparison bar chart for given names and values
#----------------------------------------------------------------
def compareBarChart(xlabel, ylabel, title, names, values, degree):
    plt.figure()
    Index = []
    for i in range(len(values)):
        Index.append(i+1)
    plt.bar(Index, values, color = 'coral')
    plt.xticks(Index, names, rotation=degree)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

#---------------------------------------------------------
# Plot number of features VS. cross-validation scores
#--------------------------------------------------------
def plotPerfScore(rfecv):    
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='coral')
    plt.title('Cross-Validation Scores of Best Chi-Squared Features')
    plt.show()

#------------------------------------------------------------------------------------------------
# Predicte if the county level cost increase/decrease for binary classification or predict the 
# increase/decrease percentage for multi-class classification for given csv. It compares the
# classification accuracy performance of a few algorithms from SKLearn package.
#--------------------------------------------------------------------------------------------------
def classification(csv_file, binary_classify = True, year = "", single_year = True, num_classes = 3):     
    data = parseCSVfile(csv_file, binary_classify, year, single_year, num_classes) 
    X_input = []
    Y_input = []
    for example in data:    
        features = example[0]
        list =[]
        for key in features:
            list.append(features[key])    
        X_input.append(list)
        Y_input.append(example[1])
   
    k=len(features)
    # calculate chi-Squared score to determine the optimal numbre of features
    try:
        ch2 = SelectKBest(chi2, k)
        X_input = ch2.fit_transform(X_input, Y_input)
    except:
        pass
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_input, Y_input, test_size=0.3, random_state=42)
   
    if binary_classify == True:
        logistic_regression = LogisticRegression(C=0.1,solver='lbfgs')
        kmean_classifier = KNeighborsClassifier(2)
    else:
        logistic_regression = LogisticRegression(C=0.1, multi_class='multinomial', solver='lbfgs')
        kmean_classifier = KNeighborsClassifier(5)
    Classifiers = [
        logistic_regression,
        MLPClassifier(solver='lbfgs'),
        kmean_classifier,
        SVC(kernel="rbf", C=0.025, probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(), 
        AdaBoostClassifier(),       
        GaussianNB()]

    Accuracy=[]
    Model=[]    
    for classifier in Classifiers:
        a = datetime.datetime.now()
        try:           
            #Feature scaling through standardization (or Z-score normalization)
            #PCA - Linear dimensionality reduction to a lower dimensional space
            std_clf = make_pipeline(StandardScaler(), PCA(), classifier)                    
            std_clf.fit(X_train, Y_train)            
            pred = std_clf.predict(X_test)                    
            b = datetime.datetime.now()            
            accuracy = accuracy_score(pred, Y_test)            
            Accuracy.append(accuracy)           
            Model.append(classifier.__class__.__name__)
            print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
            train_accuracy = std_clf.score(X_train, Y_train)
            print('Traing Accuracy of '+classifier.__class__.__name__+' is '+str(train_accuracy))
            print('Confusion Matrix of '+classifier.__class__.__name__)
            print confusion_matrix(Y_test, pred)
            print('Elapsed time = '+ str(b-a))
        except:
            pass
    if binary_classify == True:
        model = "Binary Model"
    else:
        if single_year == False:
            num_classes = 6
        model = "Multiclass Model (number of classes = " + str(num_classes) + ")"
    if single_year == True:
        title = 'Accuracies of Models for Single Year'
    else:
        title = 'Accuracies of Models for Multi-Year'
    compareBarChart(model, 'Accuracy', title, Model, Accuracy, 45)

    print("Starting to calculate the cross-validation scores for each model...")

    for classifier in Classifiers:
        try:
            rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='accuracy')
            rfecv.fit(X_train, Y_train)
            print("Optimal number of features for " +classifier.__class__.__name__+ ": %d" % rfecv.n_features_)
            plotPerfScore(rfecv)
        except:
            pass        


def main(argv):
    # parse commandline arguments    
    op = OptionParser()
    op.add_option("-B", "--binary", dest="binary_classify",
                  default=True,    
                  help="classification mode: True for binary and False for multi-class.")
    op.add_option("-S", "--single", dest="single_year",
                  default=True,    
                  help="prediction year: True for single year and False for multi-year.")   
    op.add_option("-Y", "--year", dest="year",
                  default="", type="string",
                  help="Year: 2012, 2013, ..." )
    
    op.print_help()
    (opts, args) = op.parse_args()
    opts.single_year = False
    if opts.binary_classify:
        print "Classification Mode: Binary"
    else:
        print "Classification Mode: Multi-Class"
    if opts.single_year == True:
        inputfile = "copd_single_year.csv"
    else:
        inputfile = "copd_multi_year.csv"
    classification(inputfile, opts.binary_classify, opts.year, opts.single_year)   
   
if __name__ == '__main__':
    main(sys.argv)



