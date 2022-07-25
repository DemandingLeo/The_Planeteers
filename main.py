from urllib.request import urlretrieve
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn import  metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.signal import savgol_filter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,plot_confusion_matrix,precision_score,recall_score,f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize

class Main:
    def __init__(self):
        # Imports dataset
        self.df_train = pd.read_csv('exoTrain.csv')
        self.df_train.LABEL = self.df_train.LABEL -1
        self.df_test = pd.read_csv('exoTest.csv')
        self.df_test.LABEL = self.df_test.LABEL - 1
        self.train_X,self.train_y,self.test_X,self.test_y = self.reset(self.df_train, self.df_test)

    def exoplanetStars(self):
        # Run to see samples of exoplanet stars
        fig = plt.figure(figsize=(15,40))
        for i in range(12):
            ax = fig.add_subplot(14,4,i+1)
            ax.scatter(np.arange(3197),self.df_train[self.df_train['LABEL'] == 1].iloc[i,1:],s=1)

    def nonExoplanetStars(self):
        # Run to see samples of non-exoplanet stars
        fig = plt.figure(figsize=(15,40))
        for i in range(12):
            ax = fig.add_subplot(14,4,i+1)

        ax.scatter(np.arange(3197),self.df_train[self.df_train['LABEL']==0].iloc[i,1:],s=1)

    def exoplanetHistogram(self):
        # Run to see histograms of exoplanet stars
        fig = plt.figure(figsize=(15,40))
        for i in range(12):
            ax = fig.add_subplot(14,4,i+1)
            plt.xlabel("Flux")
            plt.ylabel("Number of data points")
            self.df_train[self.df_train['LABEL']==1].iloc[i,1:].hist(bins=40)

    def nonExoplanetHistogram(self):
        # Run to see histograms of non-exoplanet stars
        fig = plt.figure(figsize=(15,40))   
        for i in range(12):
            ax = fig.add_subplot(14,4,i+1)
            plt.xlabel("Flux")
            plt.ylabel("Number of data points")
            self.df_train[self.df_train['LABEL']==0].iloc[i,1:].hist(bins=40)

    def analyze_results(model, train_X, train_y, test_X, test_y):
        """
        Helper function to help interpret and model performance.

        Args:
        model: estimator instance
        train_X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values for model training.
        train_y : array-like of shape (n_samples,)
        Target values for model training.
        test_X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values for model testing.
        test_y : array-like of shape (n_samples,)
        Target values for model testing.
    
        Returns:
        None
        """
        print("-------------------------------------------")
        print("Model Results")
        print("")
        print("Training:")
        fig = plt.figure(figsize=(22,7))
        ax = fig.add_subplot(1,3,1)
        plot_confusion_matrix(model,train_X,train_y,ax=ax,values_format = '.0f')
        plt.show()
        print("Testing:")
        fig = plt.figure(figsize=(22,7))
        ax = fig.add_subplot(1,3,1)
        plot_confusion_matrix(model,test_X,test_y,ax=ax,values_format = '.0f')
        plt.show()

    def reset(self,train,test):
        self.train_X = train.drop('LABEL', axis=1)
        self.train_y = train['LABEL'].values
        self.test_X = test.drop('LABEL', axis=1)
        self.test_y = test['LABEL'].values

    def createKNModel(self):
        self.n_neighbors = 5
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)    

    def trainModel(self):
        self.model.fit(self.train_X,self.train_y)

    def predicitonCalc(self):
        # Calculate the predictions on test_X using our trained model
 
        train_predictions = self.model.predict(self.train_X)
        test_predictions = self.model.predict(self.test_X)
        print(accuracy_score(self.train_y, train_predictions))
        print(accuracy_score(self.test_y, test_predictions))

    def analyzeModel(self):
        # Use to analyze logistical model
 
        print(self.analyze_results(model=self.model, train_X=self.train_X, train_y=self.train_y, test_X=self.test_X, test_y=self.test_y))

    def createKNModel2(self):
        # Create a model (will train later)
 
        max_iter = 1000

        self.model = LogisticRegression(max_iter=max_iter)

    def trainSecondModel(self):
        # Train the model, see accuracies, and analyze the results
 
        self.model.fit(self.train_X,self.train_y)

        train_predictions = self.model.predict(self.train_X)
        test_predictions = self.model.predict(self.test_X)
        print(accuracy_score(self.train_y, train_predictions))
        print(accuracy_score(self.test_y, test_predictions))

        self.analyze_results(model=self.model, train_X=self.train_X, train_y=self.train_y, test_X=self.test_X, test_y=self.test_y)

    def createDescisionTree(self):
        # Create a Decision Tree model (answers will vary)
 
        self.model = tree.DecisionTreeClassifier()
        
        
        self.model.fit(self.train_X,self.train_y)
        
        train_predictions = self.model.predict(self.train_X)
        test_predictions = self.model.predict(self.test_X)
        print(accuracy_score(self.train_y, train_predictions))
        print(accuracy_score(self.test_y, test_predictions))
        
        self.analyze_results(model=self.model, train_X=self.train_X, train_y=self.train_y, test_X=self.test_X, test_y=self.test_y)

    # Helper functions that we can run for the three augmentation functions that will be used
    def smote(self,a,b):
        self.model = SMOTE()
        X,y = self.model.fit_resample(a, b)
        return X,y
 
    def savgol(df1,df2):
        x = savgol_filter(df1,21,4,deriv=0)
        y = savgol_filter(df2,21,4,deriv=0)
        return x,y
    
    def fourier(df1,df2):
        train_X = np.abs(np.fft.fft(df1, axis=1))
        test_X = np.abs(np.fft.fft(df2, axis=1))
        return train_X,test_X
    
    def norm(df1,df2):
        train_X = normalize(df1)
        test_X = normalize(df2)
        return train_X,test_X
    
    def robust(df1,df2):
        scaler = RobustScaler()
        train_X = scaler.fit_transform(df1)
        test_X = scaler.transform(df2)
        return train_X,test_X

    # If we were to define these helper functions using matplotlib.pyplot we could visualize these confusion matrices and graphs. We haven't added the code to do so in this github repository for the sake of time.
