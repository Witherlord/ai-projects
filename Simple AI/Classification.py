import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier


def k_nn():
    df = pd.read_csv('teleCust1000t.csv')
    df.head()
    df['custcat'].value_counts()

    df.hist(column='income', bins=50)

    X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
            'reside']].values  # .astype(float)

    y = df['custcat'].values


    #Data Standardization give data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on distance of cases:
    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    from sklearn.neighbors import KNeighborsClassifier

    k = 4
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print(neigh)
    yhat = neigh.predict(X_test)

    from sklearn import metrics
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

    # write your code here
    k = 6
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    neigh
    yhat = neigh.predict(X_test)
    print(yhat[0:5])
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

    Ks = 10
    mean_acc = np.zeros((Ks - 1))
    std_acc = np.zeros((Ks - 1))
    ConfustionMx = [];
    for n in range(1, Ks):
        # Train Model and Predict
        neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
        yhat = neigh.predict(X_test)
        mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

        std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

    print(mean_acc)

    plt.plot(range(1, Ks), mean_acc, 'g')
    plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Nabors (K)')
    plt.tight_layout()
    plt.show()

    print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax() + 1)


def decisionTree():
    my_data = pd.read_csv("drug200.csv", delimiter=",")
    my_data[0:5]
    X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

    '''
    some features in this dataset are categorical such as Sex or BP. 
    Unfortunately, Sklearn Decision Trees do not handle categorical variables. 
    But still we can convert these features to numerical values. 
    pandas.get_dummies() Convert categorical variable into dummy/indicator variables.
    '''

    from sklearn import preprocessing
    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F', 'M'])
    X[:, 1] = le_sex.transform(X[:, 1])

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
    X[:, 2] = le_BP.transform(X[:, 2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit(['NORMAL', 'HIGH'])
    X[:, 3] = le_Chol.transform(X[:, 3])

    y = my_data["Drug"]

    from sklearn.model_selection import train_test_split

    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

    '''
    We will first create an instance of the DecisionTreeClassifier called drugTree.
    Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
    '''

    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    drugTree  # it shows the default parameters

    drugTree.fit(X_trainset, y_trainset)

    predTree = drugTree.predict(X_testset)

    print(predTree[0:5])
    print(y_testset[0:5])

    from sklearn import metrics
    import matplotlib.pyplot as plt
    print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

    '''
    To get a nice picture of your decision tree
    '''

    # Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
    # !conda install -c conda-forge pydotplus -y
    # !conda install -c conda-forge python-graphviz -y

    from sklearn.externals.six import StringIO
    import pydotplus
    import matplotlib.image as mpimg
    from sklearn import tree
    #% matplotlib inline

    dot_data = StringIO()
    filename = "drugtree.png"
    featureNames = my_data.columns[0:5]
    targetNames = my_data["Drug"].unique().tolist()
    out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data,
                               class_names=np.unique(y_trainset), filled=True, special_characters=True, rotate=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(100, 200))
    plt.imshow(img, interpolation='nearest')


def logisticRegression():
    import pandas as pd
    import pylab as pl
    import numpy as np
    import scipy.optimize as opt
    from sklearn import preprocessing
    import matplotlib.pyplot as plt

    churn_df = pd.read_csv("ChurnData.csv")
    churn_df.head()

    churn_df = churn_df[
        ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
    churn_df['churn'] = churn_df['churn'].astype('int')
    churn_df.head()

    X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

    y = np.asarray(churn_df['churn'])

    # Also, we normalize the dataset:

    from sklearn import preprocessing
    X = preprocessing.StandardScaler().fit(X).transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    # Regularization is a technique used to solve the overfitting problem in machine learning models.
    # C parameter indicates inverse of regularization strength which must be a positive float.
    # Smaller values specify stronger regularization.

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    LR = LogisticRegression(C=0.02, solver='lbfgs').fit(X_train, y_train)

    yhat = LR.predict(X_test)

    yhat_prob = LR.predict_proba(X_test)    # predict_proba returns estimates for all classes,

    from sklearn.metrics import jaccard_similarity_score
    print(jaccard_similarity_score(y_test, yhat))

    # confusion matrix

    from sklearn.metrics import classification_report, confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

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

    print(confusion_matrix(y_test, yhat, labels=[1, 0]))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Confusion matrix')

    print(classification_report(y_test, yhat))

    from sklearn.metrics import log_loss
    log_loss(y_test, yhat_prob)


def supportVectorMachine():
    import pandas as pd
    import pylab as pl
    import numpy as np
    import scipy.optimize as opt
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    cell_df = pd.read_csv("cell_samples.csv")
    cell_df.head()

    ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue',
                                                   label='malignant');
    cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign',
                                              ax=ax);
    plt.show()

    print(cell_df.dtypes)

    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

    print(cell_df.dtypes)

    feature_df = cell_df[
        ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(feature_df)

    cell_df['Class'] = cell_df['Class'].astype('int')
    y = np.asarray(cell_df['Class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    print('Train set:', X_train.shape, y_train.shape)
    print('Test set:', X_test.shape, y_test.shape)

    from sklearn import svm
    clf = svm.SVC(kernel='linear', gamma='auto')
    clf.fit(X_train, y_train)

    yhat = clf.predict(X_test)

    # Evaluation

    from sklearn.metrics import classification_report, confusion_matrix
    import itertools

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

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

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
    np.set_printoptions(precision=2)

    print(classification_report(y_test, yhat))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title='Confusion matrix')

    # or f1_score

    from sklearn.metrics import f1_score
    print(f1_score(y_test, yhat, average='weighted'))

    # or jaccard index for accuracy

    from sklearn.metrics import jaccard_similarity_score
    print(jaccard_similarity_score(y_test, yhat))


if __name__ == '__main__':
    # k_NN()
    # decisionTree()
    # logisticRegression()
    supportVectorMachine()


    row = list()
    row.append()
