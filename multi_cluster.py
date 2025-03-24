import sys
import re
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader
from pyspark.sql import Row
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, FloatType, DateType, ArrayType
from py4j.java_gateway import JavaGateway
from pyspark.sql.functions import split, array
from pyspark.ml.functions import array_to_vector
from pyspark.ml.clustering import KMeans
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QLabel, QWidget, QInputDialog, QGridLayout


class MultiCluster:
    '''
        A class for clustering on multiple features
    '''

    #The index at which the attributes for clustering start
    ATTRIBUTE_START = 2
    
    def __init__(self, external_relation, fig:plt.figure):
        '''
            Constructor for the MultiCluster class. Takes an external relation which is the tags we will
            cluster on. Initializes some variables for use later and they will be assigned as needed.

            Parameter:
                external_relation - the dataframe with the appID, name, and other columns to cluster on
        '''
        #the tags that are passed in from the external call
        self.external_relation = external_relation
        #the dataframe as a collection
        self.k_collection = None
        #the spark instance
        self.spark = None
        #the dataframe for the unique tags on steam
        self.tags = {}
        #the scanner and sysin for user input
        self.scanner = None
        self.sys_in = None
        self.figure = fig


    def do_clustering(self, with_features):
        '''
            Performs the clustering algorithm on the dataframe that was made in make_composite_table()

            Parameters:
                with_features - the dataframe with the features column, structured properly.
        '''

        #get number of clusters
        print("Enter how many clusters you want to have: ")
        k, done1 = QInputDialog.getInt(None, "K-Means Clustering", "Choose Number of Clusters:\n", 1, 0, 7)
        
        try:
            #try to conver k to an int
            k = int(k)
            #make the k-means
            kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features_vec")
            #train the model
            model = kmeans.fit(with_features)
            #fit the data to the model
            final_data = model.transform(with_features)
            #show all the clusters
            for i in range(0, k):
                final_data.where(final_data.prediction == i).show()
            

            print("Would you like to see a graph of the clusters?(y/n)\nGraph coloring is limited to 7 clusters.")
            choices = ["Yes", "No"]
            graph_choice, done1 = QInputDialog.getItem(None, "K-Means Clustering", "Print?:\n", choices, 0)
            if graph_choice == "Yes" and done1:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                colors = ["bo", "go", "ro", "co", "mo", "yo", "ko"]
                for i in range(0, k):
                    for game in [row.features_vec for row in final_data.where(final_data.prediction == i).collect()[0:20]]:
                        pointsx = []
                        pointsy = []
                        pointsx.append(game[0])
                        if len(game) >= 2:
                            pointsy.append(game[1])
                        else:
                            pointsy.append(game[0])
                        ax.plot(pointsx, pointsy, colors[i])

        #value error if anything went wrong in the try block
        except ValueError as ve:
            print("ValueError: Number of clusters could not be converted to integer")


    def make_tags_dictionary(self):
        '''
            Makes a dictionary for the tags so that the names of the tags can be used as the keys to get the values
        '''
        #gets the filename
        print("Enter the file name that contains the tags: ")
        user_input = self.scanner(self.sys_in).nextLine()
        
        #tries to open and read from the file
        try:
            file = open(user_input)
            line = file.readline()
            #while there are still lines to read
            while line:
                #split and clean the input
                line = line.split(";")
                line[1] = line[1].strip()
                #add the key value pair to the dictionary and advance the line
                self.tags.update({line[1]: int(line[0])})
                line = file.readline()
            file.close()
        except IOError:
            print("ERROR: File for tags could not be opened or read from\n")
            print("SUPPLIED: ", user_input)

    def append_tags(self):
        '''
            Helper method for appending the tags
        '''
        max_size = 0
        #use the external tags to create a relation that has a vector that has all the tags as integers
        array_col = 2
        #takes the external tags and makes an array, then makes an array of arrays
        split_data = self.external_relation.select("appID", "name", split(self.external_relation.tags, ',').alias('split_tags')).collect()
        #for each of the tasg sub-arrays replace the colon and everything after with nothing, so just the tags are left
        for i in range(len(split_data)):
            arr = split_data[i][array_col]
            for j in range(len(arr)):
                #clean the tags entries
                arr[j] = re.sub("[\"]*", "", arr[j])
                arr[j] = re.sub(":.*", "", arr[j])
                arr[j] = arr[j].strip()
                #replace the tags with their numerical id
                arr[j] = self.tags[arr[j]]
                if len(arr) > max_size:
                    max_size = len(arr)

        #gets the index of the features list
        arr_index = len(self.k_collection[0]) - 1
        #pad the array with 0's for the length of the longest array in the current collection
        for i in range(len(split_data)):
            arr = split_data[i][array_col]
            for j in range(max_size - len(arr)):
                arr.append(0)
            self.k_collection[i][arr_index].extend(arr)


    def make_features_col(self):
        '''
            Appends to the features array the values for the desired column
        '''
        #makes a list of the names of the columns and saves the last index for the array of features
        cols = self.external_relation.columns
        arr = len(self.k_collection[0]) - 1
        #remove the first two elements of the array since they are just the appID and name
        cols = cols[self.ATTRIBUTE_START:]
        #adds elements to the features array depending on which column it belongs to
        for i in range(len(cols)):
            if cols[i] == 'price':
                for j in range(len(self.k_collection)):
                    self.k_collection[j][arr].append(self.k_collection[j].price)
            if cols[i] == 'positive_reviews':
                for j in range(len(self.k_collection)):
                    self.k_collection[j][arr].append(self.k_collection[j].positive_reviews)
            if cols[i] == 'negative_reviews':
                for j in range(len(self.k_collection)):
                    self.k_collection[j][arr].append(self.k_collection[j].negative_reviews)
            if cols[i] == 'tags':
                self.append_tags()
        

    def receive_instances(self, spark_session, scanner, sys_in):
        '''
            Gets the instances of the SparkSession, Scanner, and input from the main class

            Parameters:
                spark_session - the SparkSession instance
                scanner - the Scanner to take user input
                sys_in - System.in for taking user input
        '''
        #spark session from main file
        self.spark = spark_session
        #Scanner for user input
        self.scanner = scanner
        #Standard input to accept user inputj using the scanner
        self.sys_in = sys_in
        

    def go(self):
        '''
            Controls the main flow of logic for the MultiClustering class. Adds the features
            column to the dataframe and calls methods needed to append to it.
        '''
        #adds a features array that will be converted to a vector
        self.external_relation = self.external_relation.withColumn('features', array())
        #make a save a reference to the relation as a collection
        self.k_collection = self.external_relation.collect()
        if 'tags' in self.external_relation.columns:
            #makes the tags dictionary if it's needed for the k-means
            self.make_tags_dictionary()
        #makes the features column
        self.make_features_col()
        #makes a new dataframe from the updated collection and gets the appID, name, and features_vector
        with_features = self.spark.createDataFrame(self.k_collection)
        with_features = with_features.select('appID', 'name', array_to_vector('features').alias('features_vec'))
        #does the clustering
        self.do_clustering(with_features)