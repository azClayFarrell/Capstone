###
# Clay Farrell
# Capstone Project
# 12/14/23
# The file handling the K-Means Clustering Algorithm
###


import matplotlib.pyplot as plt
from pyspark.sql.functions import array
from pyspark.ml.functions import array_to_vector
from pyspark.ml.clustering import KMeans


class MultiClusterGUI:
    '''
        A class for clustering on multiple features
    '''

    #The index at which the attributes for clustering start
    ATTRIBUTE_START = 2

    def __init__(self, fig:plt.figure):
        '''
            Constructor for the MultiCluster class. Takes an external relation which is the tags we will
            cluster on. Initializes some variables for use later and they will be assigned as needed.

            Parameter:
                fig : a pyplot.figure we use to draw our data to
        '''
        #the dataframe that is passed in from the main controller
        self.external_relation = None
        #the dataframe as a collection
        self.k_collection = None
        #the spark instance
        self.spark = None
        #the figure that is used for graphing
        self.figure = fig
        #the amount of clusters we are going to make
        self.cluster_amt = None
        #the axes we are graphing on
        self.axes = None
        #the actual graph data
        self.plot_graphs = []
        #the dictionary that is going to be used for annotations
        self.annotation_dict = {}
        #the results of the cluster algorithm as a string
        self.cluster_string = None
        #the x and y axes that we are using to graph the data points to the figure
        self.x_and_y_axes = None


    def set_df_and_cluster_amt_and_xy(self, data, cluster_amt, x_and_y_axes):
        '''
            Takes in a dataframe, int for the amount of clusters, and the names of the x and y axes options
            and assigns them to variables

            Parameters:
                data: a relation passed in to be used for the clustering algorithm
                cluster_amt: the number of clusters the program needs to make for the output graph
                x_and_y_axes: an array for what the x and y axes could be
        '''
        self.external_relation = data
        self.cluster_amt = cluster_amt
        self.x_and_y_axes = x_and_y_axes


    def get_axes_and_artist_and_annotations(self):
        '''
            Returns an array of the axes, graph, and annotations dictionary

            Returns:
                An array containing the axes, graph, and annotations dictionary
        '''
        return [self.axes, self.plot_graphs, self.annotation_dict]
    

    def get_output_string(self):
        '''
            Returns the string for the output of the clustering algorithm such that it is intercepted to be put into
            text for the GUI

            Returns:
                cluster_string: the string that represents the clustering results from K-means
        '''
        return self.cluster_string


    def get_showString(self, df, n=20, truncate=True, vertical=False):
        '''
            Does the same underlying code as the .show() function but intercepts it before print is called so that
            we can redirect this to a label in the GUI

            Parameters:
                df: the dataframe we are getting the string for
                n : int, optional
                    Number of rows to show.
                truncate : bool or int, optional
                    If set to ``True``, truncate strings longer than 20 chars by default.
                    If set to a number greater than one, truncates long strings to length ``truncate``
                    and align cells right.
                vertical : bool, optional
                    If set to ``True``, print output rows vertically (one line
                    per column value).

            Returns:
                the string version of the dataframe based on the values given for command line arguments

            credit to: pault on stack overflow for the example. Not all the code is needed for this project but is being
                    kept for extensibility
                Link - https://stackoverflow.com/questions/55653609/saving-result-of-dataframe-show-to-string-in-pyspark
        '''
        default_truncate = 20
        if isinstance(truncate, bool) and truncate:
            return df._jdf.showString(n, default_truncate, vertical)
        else:
            return df._jdf.showString(n, int(truncate), vertical)


    def do_clustering(self, with_features):
        '''
            Performs the clustering algorithm on the dataframe that was made in make_composite_table()

            Parameters:
                with_features : the dataframe with the features column, structured properly.
        '''        
      
        #make the k-means
        kmeans = KMeans().setK(self.cluster_amt).setSeed(1).setFeaturesCol("features_vec")
        
        #train the model
        model = kmeans.fit(with_features)
        #fit the data to the model
        final_data = model.transform(with_features)

        #print all the clusters to command line
        self.cluster_string = ""
        for i in range(0, self.cluster_amt):
            self.cluster_string += self.get_showString(final_data.where(final_data.prediction == i)) + "\n\n"

        print(self.cluster_string)

        #for graphing to the main window after running
        self.graph_to_main_window(final_data)

    def graph_to_main_window(self, final_data):
        '''
            Takes the final data clusters and plots their elements to the graph

            Parameters:
                final_data: the clusters to be graphed
        '''
        dollar_scale = 100
        precission = 2
        x_axis_name_idx = 0
        y_axis_name_idx = 1

        #make sure to clear the graph references out before starting or you could have stale ones on method start
        self.plot_graphs.clear()
        #make the colors and get the labels for the graph from the dataframe that is passed in
        features_labels = self.external_relation.columns
        features_labels = features_labels[self.ATTRIBUTE_START:-1]

        #finds the indecies for the x and y axis that the user wants to use for graphing
        x_axis_index = features_labels.index(self.x_and_y_axes[x_axis_name_idx])
        y_axis_index = features_labels.index(self.x_and_y_axes[y_axis_name_idx])

        #clear the current figure so we dont overdraw everything and make a new one
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.axes = ax

        self.make_annotation_dictionary(final_data, x_axis_index, y_axis_index)

        #iterate through the clusters and graph with colors based on cluster number
        for i in range(0, self.cluster_amt):
            pointsx = []
            pointsy = []
            for game in [row.features_vec for row in final_data.where(final_data.prediction == i).collect()[0:20]]:
                # appends the x and y points using their axes index to the arrays to have them graphed, adjusting the 
                # scale for price
                if self.x_and_y_axes[x_axis_name_idx] == "price":
                    pointsx.append(round(game[x_axis_index]/dollar_scale, precission))
                else:
                    pointsx.append(game[x_axis_index])

                if self.x_and_y_axes[y_axis_name_idx] == "price":
                    pointsy.append(round(game[y_axis_index]/dollar_scale, precission))
                else:
                    pointsy.append(game[y_axis_index])

            # makes new scatter plots and appends the references to them to the end of the list of graphs
            self.plot_graphs.append(ax.scatter(x=pointsx, y=pointsy))

        #set labels, titles, and legend back
        ax.set_xlabel(features_labels[x_axis_index])
        ax.set_ylabel(features_labels[y_axis_index])
        ax.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7'])
        ax.set_title("K-Means Clustering Graph")


    def make_annotation_dictionary(self, final_data, x_axis_index, y_axis_index):
        '''
            Makes a dictionary of dictionary with 2d arrays that have the information for the appID and name of the game
            you are pointing at in the graph. The keys are the strings versions of the first and second feature of the
            features vector of the data the dictionary is intended to be returned out of this file to the one containing
            the graph that needs the information

            Parameters:
                final_data: the data after the k-means clustering algorithm has been run
                x_axis_index: the index for the x axis in the features column in the final_data dataframe
                y_axis index: the index for the y axis in the features column in the final_data dataframe
        '''
        # if there were two or more dimensions(features) that we ran the clustering algorithm on
        two_or_more_features = 2
        # the first twenty entries per cluster
        first_20 = 20
        for i in range(0, self.cluster_amt):
            #for each cluster and for each game in that cluster
            games_in_cluster_as_rows = final_data.where(final_data.prediction == i).collect()[:first_20]
            for game in games_in_cluster_as_rows:
                #check how long the features array is and determine what keys and values should be inserted into the dictionary
                if len(game.features_vec) >= two_or_more_features:
                    if str(game.features_vec[x_axis_index]) not in self.annotation_dict:
                        #if there is no key yet for the x value then there can be no dictionary for the y value
                        self.annotation_dict.update({str(game.features_vec[x_axis_index]): \
                                                {str(game.features_vec[y_axis_index]): [[str(game.appID), game.name]]}})

                    elif str(game.features_vec[y_axis_index]) not in self.annotation_dict[str(game \
                                                                                        .features_vec[x_axis_index])]:
                        #if the key for y not inside the dictionary that is in the annotations_dict for x
                        self.annotation_dict[str(game.features_vec[x_axis_index])].update \
                            ({str(game.features_vec[y_axis_index]): [[str(game.appID), game.name]]})
                    else:
                        #the last case is that the x and y keys exist and we just need to append this new game entry
                        self.annotation_dict[str(game.features_vec[x_axis_index])] \
                            [str(game.features_vec[y_axis_index])].append([str(game.appID), game.name])
                else:
                    if str(game.features_vec[x_axis_index]) not in self.annotation_dict:
                        #if there is no key yet for the x value then there can be no dictionalry for the y value
                        self.annotation_dict.update({str(game.features_vec[x_axis_index]): \
                                                {str(game.features_vec[x_axis_index]): [[str(game.appID), game.name]]}})

                    elif str(game.features_vec[x_axis_index]) not in self.annotation_dict[str(game \
                                                                                        .features_vec[x_axis_index])]:
                        #if the key for y not inside the dictionary that is in the annotations_dict for x
                        self.annotation_dict[str(game.features_vec[x_axis_index])].update \
                            ({str(game.features_vec[x_axis_index]): [[str(game.appID), game.name]]})
                    else:
                        #the last case is that the x and y keys exist and we just need to append this new game entry
                        self.annotation_dict[str(game.features_vec[x_axis_index])] \
                            [str(game.features_vec[x_axis_index])].append([str(game.appID), game.name])


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
        

    def set_spark_session(self, spark_session):
        '''
            Gets the instances of the SparkSession so we can remake the dataframe at the end

            Parameters:
                spark_session : the SparkSession instance
        '''
        #spark session from main file
        self.spark = spark_session
        

    def go(self):
        '''
            Controls the main flow of logic for the MultiClustering class. Adds the features
            column to the dataframe and calls methods needed to append to it.
        '''
        #adds a features array that will be converted to a vector
        self.external_relation = self.external_relation.withColumn('features', array())
        #make a save a reference to the relation as a collection
        self.k_collection = self.external_relation.collect()
        
        #makes the features column
        self.make_features_col()
        #makes a new dataframe from the updated collection and gets the appID, name, and features_vector
        with_features = self.spark.createDataFrame(self.k_collection)
        with_features = with_features.select('appID', 'name', array_to_vector('features').alias('features_vec'))
        #does the clustering
        self.do_clustering(with_features)