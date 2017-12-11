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
from util import *
import datetime
from optparse import OptionParser


def parseCSVfile(csv_file, binary_classify, year, single_year):
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        records = list(reader)
    data = []
    keys = records[0] #use the header of csv file as the keys
    cols = len(keys)
    
    startIndex = 4
    for j in range(startIndex, cols-1):
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
##                    if value < nsum:
##                        y = -1                
##                    elif value < psum:
##                        y = 0
##                    else:
##                        y = 1
                
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
    print len(data)
    return data


#-------------------------------------------------------------
# Using SGD algorithm with eta = 0.01 do the linear prediction
#-------------------------------------------------------------
def learnPredictor(trainExamples, testExamples, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.
    '''
    weights = {}  # feature => weight
    for i in range(0, numIters):
        for trainExample in trainExamples:
            x = trainExample[0]
            y = trainExample[1]           
            features = x            
            for k in features:
                if k not in weights:
                    weights[k] = 0
            d = 1 - sum(weights[k] * features[k] for k in features) * y                        
            for k in features:
                if d > 0:
                    gradient_loss = - features[k] * y
                else:
                    gradient_loss = 0               
                weights[k] = weights[k] - eta * gradient_loss
        predictor = lambda(x) : (1 if dotProduct(x, weights) >= 0 else -1)        
        trainLoss = evaluatePredictor(trainExamples, predictor)
        testLoss = evaluatePredictor(testExamples, predictor)
    print "Iteration %d: training error = %f, test error = %f" % (i, trainLoss, testLoss)
    return weights

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

#---------------------------------------------------------------------------------------------
# Predicte if the county level cost increase/decrease for binary classification or predict the 
# increase/decrease percentage for multi-class classification for given csv. It compares the
# classification accuracy performance of a few algorithms from SKLearn package.
#---------------------------------------------------------------------------------------------
def classification(csv_file, binary_classify = True, year = "", single_year = True):     
    data = parseCSVfile(csv_file, binary_classify, year, single_year) 
    if binary_classify == True: #baseline model using SGD
        train_size = (int)(0.7*len(data))
        trainExamples = data[:train_size]
        testExamples = data[train_size:]
        weights = learnPredictor(trainExamples, testExamples, numIters=20, eta=0.01)
        print "Weights = ", weights

        example = data[0]
        features = example[0]
        names = []
        for feature in features:
            names.append(feature)
        values = []
        for key in weights:
            values.append(weights[key])
        compareBarChart('Feature', 'Weight', 'Weights of Features', names, values, 90)


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
            std_clf = make_pipeline(StandardScaler(), classifier)
            std_clf.fit(X_train, Y_train)            
            pred = std_clf.predict(X_test)                    
            b = datetime.datetime.now()            
            accuracy = accuracy_score(pred, Y_test)            
            Accuracy.append(accuracy)
            Model.append(classifier.__class__.__name__)
            print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
            train_accuracy = std_clf.score(X_train, Y_train)
            print('Traing Accuracy of '+classifier.__class__.__name__+' is '+str(train_accuracy))
            print('Elapsed time = '+ str(b-a))
        except:
            pass
    if binary_classify == True:
        model = "Binary Model"
    else:
        model = "Multiclass Model"
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
    op.add_option("-Y", "--year", dest="year",
                  default="", type="string",
                  help="Year: 2012, 2013, ..." )
    
    op.print_help()
    (opts, args) = op.parse_args()
    if opts.binary_classify:
        print "Classification Mode: Binary"
    else:
        print "Classification Mode: Multi-Class"        
    classification("copd_perc_all.csv", True, "", True)
    #classification("copd_merge_2012-2015.csv", False, 0, "")

if __name__ == '__main__':
    main(sys.argv)



