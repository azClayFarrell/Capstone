###
# Clay Farrell
# Capstone Project
# 12/14/23
# The file handling the Linear Regression Algorithm
###


from matplotlib import pyplot
import numpy as np

from pyspark.sql import functions


class GamesRegressionGUI:
    '''
        Handles the regression algorithm and graph making for the linear regression of steam games
    '''

    def __init__(self, fig:pyplot.figure):
        '''
            Constructor for the Regression_Algo class. Makes a spark session and variables to use for user input. Also
            makes dataframes that are defaulted to None.

            Parameter:
                fig : a pyplot.figure we use to draw our data to
        '''

        #The dataframes we use. One that was passed in and the others to be made
        self.external_data = None

        #figure for graphing
        self.figure = fig
        #the axes for graphing
        self.axes = None
        #the actual graph
        self.regression_plot = None

    def receive_dataframe(self, data):
        '''
            Takes in a dataframe and assigns it to the dataframe for linear regression

            Parameters:
                data : a relation passed in to be used for the linear regression algorithm
        '''
        self.external_data = data
        

    def alt_linear_regression(self):
        '''
            Does the linear regression for the data points for the figure that was passed in. Specifically we do LR for
            the years that we have data for in the dataset and the number of games that were published in that year.
        '''
        #the minimum and maximum years that a game could be published
        min_year = 1996
        max_year = 2023

        years, published_games = self.alt_make_data()
        
        self.figure.clear()

        #axes and plotting of the data in a way the GUI will understand
        ax = self.figure.add_subplot(111)
        
        self.axes = ax

        #make the titles and labels again
        ax.set_title("Linear Regression Graph")
        ax.set_xlabel("Year")
        ax.set_ylabel("# of Games Published")

        #adjust the x bound so that the entries with incorrect date format are not shown
        ax.set_xbound(min_year, max_year)
        # Add scatterplot
        self.regression_plot = ax.scatter(years, published_games, s=60, alpha=0.7, edgecolors="k")

        # Fit linear regression via least squares with numpy.polyfit
        # It returns an slope (b) and intercept (a)
        # deg=1 means linear fit (i.e. polynomial of degree 1)
        m, b = np.polyfit(years, published_games, deg=1)

        # Create sequence of 100 numbers (100 was in the example) for making the regression line
        # seems to break if num is too low
        xseq = np.linspace(min_year + 1, max_year, num=100)
        # Plot regression line
        ax.plot(xseq, m * xseq + b, color="k", lw=2.5)


    def get_axes_and_artist(self):
        '''
            Returns a list that contains the axes of the graph and a reference to the PathCollection object that makes
            up the graph (Artist). Both are needed for doing annotations outside of this class.

            Returns:
                a list containing the axes of the graph and a reference to the PathCollection object that we use to
                draw annotations on (it is a subclass of Artist)
        '''
        return [self.axes, self.regression_plot]


    def alt_make_data(self):
        '''
            Goes through all the years that games were published in the dataset and makes two lists. One for all the
            years (1997 - 2022 for the dataset) and another for the amount of games that were released in that year on
            Steam. So, for each index in the lists, there will be a year and a number that is associated with one
            another. These two data points are what we use for the Linear Regression

            Returns:
                a list containing two lists within which correspond to the years that games were released in the dataset
                and another that corresponds to how many games were released for that year.
        '''
        
        max_year = self.external_data.agg(functions.max(self.external_data.release_date)).collect()[0][0]
        min_year = self.external_data.agg(functions.min(self.external_data.release_date)).collect()[0][0]
        years = []
        published_games = []
        for i in range(min_year, max_year + 1):
            years.append(i)
            published_games.append(self.external_data.where(self.external_data.release_date == i).count())
        return [years, published_games]