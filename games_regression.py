import sys
import re
from matplotlib import pyplot
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, FloatType, DateType
from py4j.java_gateway import JavaGateway


class GamesRegression:
    '''
        Handles the regression algorithm and graph making for the linear regression of steam games
    '''

    #The schema for the Linear Regression dataset
    #X IS THE YEAR, Y IS THE PRICE
    SCHEMA = StructType([StructField("release_date", DateType(), False), StructField("price", IntegerType(), False)])

    def __init__(self, _scanner, _sys_in, data, fig:pyplot.figure):
        '''
            Constructor for the Regression_Algo class. Makes a spark session and variables to use for user input. Also
            makes dataframes that are defaulted to None.
        '''
        #Scanner for user input
        self.scanner = _scanner
        #Standard input to accept user inputj using the scanner
        self.sys_in = _sys_in
        #The dataframes we use. One that was passed in and the others to be made
        self.external_data = data
        self.dataframe_training = None
        self.dataframe_test = None
        self.figure = fig
    

    def make_dataframes(self):
        '''
            Makes dataframes for the training and testing of the LR model using a 30/70 random split
            of the data that is passed in externally in the constructor
        '''
        #random 30% of the data in element 0, the remaining 70% in element 1
        random_gen_dataframes = self.external_data.randomSplit([0.3, 0.7])
        self.dataframe_training = random_gen_dataframes[0]
        self.dataframe_test = random_gen_dataframes[1]
        ###DEBUG###
        print("Number of rows in TRAINING: " + str(self.dataframe_training.count()))
        print("Number of rows in TEST: " + str(self.dataframe_test.count()))


        

    def linear_regression(self):
        #makes the training dataframe

        #gets the column for the release_date
        features_col = self.dataframe_training.columns[:-1]
        #add that to a vector assembler
        vect_assembler = VectorAssembler(inputCols = features_col, outputCol = "features", handleInvalid = "skip")
        #append the features column to the training dataset
        data_with_features = vect_assembler.transform(self.dataframe_training)
        data_with_features.show()
        #then select only the features and the y coordinate column
        training = data_with_features.select("features", "price")
        regression = LinearRegression(featuresCol = "features", labelCol = "price")
        model = regression.fit(training)


        #makes the testing dataframe

        #gets the column for the release_date
        features_col = self.dataframe_test.columns[:-1]
        #add that to a vector assembler
        vect_assembler = VectorAssembler(inputCols = features_col, outputCol = "features", handleInvalid = "skip")
        #append the features column to the test dataset
        data_with_features = vect_assembler.transform(self.dataframe_test)
        #then select only the features and the y coordinate column
        testing = data_with_features.select("features", "price")
        #makes a prediction model from the test data
        predict = model.evaluate(testing)
        predict.predictions.show()
        print("=======================================================================")
        print("The coefficient of the model is: ", str(model.coefficients))
        print("\nThe intercept of the model is: ", str(model.intercept))
        evaluation = RegressionEvaluator(labelCol = "price", predictionCol = "prediction")
        #shows how good the model fit the data
        print("\nLevel of confidence in the model (closer to 1 is better): ", evaluation.evaluate(predict.predictions,
                                                                                        {evaluation.metricName: "r2"}))
        self.show_graph(model.intercept, model.coefficients)

    def show_graph(self, intercept, slope):
        self.figure.clear()
        date_axis = []
        price_axis = []
        #years = []
        #for num in range(19960101, 20231231):
        #     years.append(num)
        data_collection = self.external_data.collect()
        for i in range(len(data_collection)):
            date_axis.append(data_collection[i][0])
            price_axis.append(data_collection[i][1])
        price_max = max(price_axis)
        price_min = min(price_axis)
        #something is up with the years being VERY OLD
        #date_max = max(date_axis)
        #date_min = min(date_axis)

        #axes and plotting of the data in a way the GUI will understand
        ax = self.figure.add_subplot(111)
        ax.plot(date_axis, price_axis, "bo")

        #adjust the x bound so that the entries with incorrect date format are not shown
        ax.set_xbound(19960101, 20231231)
        
        #pyplot.plot(slope * years + intercept, linestyle = 'solid')
        #TODO find out why the years go back so far and fix it
        # pyplot.axis([19960101, 20231231, price_min, price_max])
        # pyplot.show()
    
    def do_option(self, user_input):
        match user_input:
            case "1":
                #self.dataframe_training.show()
                #self.dataframe_test.show()
                self.linear_regression()
                
            case "0":
                return
            case _:
                print("ERROR: Invalid Input")

    def main_menu(self):
        self.make_dataframes()
        #if both dataframes generate correctly then give the option for LR model to be made
        if self.dataframe_training != None and self.dataframe_test != None:
            
            self.linear_regression()
            # user_input = 0
            #while the user doesn't want to exit take input and pass it to do_option
            # while user_input != "0":
            #     print("\nEnter an option\n--------------------------"\
            #         "\n1: Linear Regression & Graph\n" \
            #             "0: Exit\n")
            #     user_input = self.scanner(self.sys_in).nextLine()
            #     self.do_option(user_input)
        else:
                    #print an error message and tell the user which one(s) of the dataframes was not generated
                    print("ERROR: One or both of the dataframes for training or testing is not generated")
                    print("TRAINING dataframe is None: ", str(self.dataframe_training == None))
                    print("TESTING dataframe is None: ", str(self.dataframe_test == None))
                    print("Returning to steam_games.py...")
        