###
# Clay Farrell
# Capstone Project
# 12/14/23
# The entry point of the program in control of making the GUI elements
###


import sys
import games_regression_GUI
import multi_cluster_GUI
import steam_games_GUI

# importing various GUI libraries
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QLabel, QWidget, QGridLayout, QHBoxLayout, \
                            QCheckBox, QMainWindow, QSpinBox, QComboBox, QDockWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

#The usage message for this program
USAGE = "\n=========================================================================================\n" \
        "Usage: [path to spark-submit script] [path to steam_games.py] [path to csv steam file]\n" \
        "=========================================================================================\n"

#the needed number of command line arguments
MIN_ARGS = 2

class MainMenu(QMainWindow):
    '''
        Makes the main menu for the program and bridges any gaps that are present to hook up the forward facing GUI to
        the underlying logic
    '''

    COLUMN_NAMES = ["App ID", "App Name", "Description", "Developer", "Publisher", "Genre", "Tags", "Type",
                    "Categories","Owners","Positive Reviews","Negative Reviews","Price","Initial Price","Discount",
                    "CCU","Languages","Platforms","Release Date","Required Age","Website","Header Image","Select All"]
    

    COLUMN_VALUES = ["appID","name","short_description","developer","publisher","genre","tags","type","categories",
                     "owners","positive_reviews","negative_reviews","price","initial_price","discount","ccu",
                     "languages","platforms","release_date","required_age","website","header_image", "*"]

    #indecies of important data
    FIRST_GAME_IDX = 0
    APPID = 0
    DATAPOINT_IDX = 0
    APPNAME = 1

    #min and max clusters
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 7

    def __init__(self, parent=None):
        '''
            Constructor for the MainMenu class. Creates a SparkSession using the local specifics and
            names it. Initializes the scanner and input variables to take user input. Initializes the
            main dataframe.

            Parameters:
                parent: the parent widget of this widget, defaults to None in which case this window will be free floating
        '''
        super(MainMenu, self).__init__(parent)

        #makes a steam games logic driver so that we can use the functionality of actually using spark
        self.games_controller = steam_games_GUI.SteamGamesLogicDriver(self, sys.argv[1])

        #makes the GUI
        self.generate_GUI()
        
        #everything below here is stuff for making annotations on the graph

        #make a new games regression object
        self.regression = games_regression_GUI.GamesRegressionGUI(self.regression_figure)
        self.regression_axes = None
        self.regression_annotations = None
        self.regression_plot = None
        self.regression_figure.canvas.mpl_connect('motion_notify_event', self.regression_on_hover)

        #create an instance of the MultiCluster class
        self.multi = multi_cluster_GUI.MultiClusterGUI(self.k_means_figure)
        self.k_means_axes = None
        self.k_means_annotations = None
        self.k_means_plots = None
        self.k_means_annotations_dict = None
        self.k_means_figure.canvas.mpl_connect('motion_notify_event', self.k_means_on_hover)


    def generate_GUI(self):
        '''
            Creates all the elements and calls all the functions for making the GUI for the program
        '''
        #the columns that have currently been selected with the check boxes
        self.column_selection = []

        #GUI elements for layouts
        main_layout = QVBoxLayout()
        ribbon_layout = QHBoxLayout()
        body_layout = QHBoxLayout()
        graphs_layout = QVBoxLayout()
        options_menu_layout = QVBoxLayout()
        options_button_layout = QHBoxLayout()
        k_means_options_layout = QHBoxLayout()
        boxes_layout = QGridLayout()

        #the widgets that will be nested into one another to make the main layout of the menu
        ribbon  = QWidget()
        body = QWidget()
        graphs = QWidget()
        options = QWidget()
        k_means_options = QWidget()
        box_selection_widget = QWidget()
        box_dock_widget = QDockWidget()
        menu_option_buttons = QWidget()

        #makes the checkboxes and saves a list of references to the boxes
        self.box_references = []
        for i in range(len(self.COLUMN_NAMES) + 1):
            self.make_checkboxes(i, boxes_layout)

        #sets the boxes in their grid layout
        box_selection_widget.setLayout(boxes_layout)

        #make the Dock widget for the check boxes unable to be closed by not including it in the features
        box_dock_widget.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        #make the internal widget for the dock widget the check box widget
        box_dock_widget.setWidget(box_selection_widget)

        #makes the buttons for the menu options like printing, k-means, LinearRegression, and making the dataframe
        self.make_menu_buttons(options_button_layout)
        self.make_k_means_options(k_means_options_layout)
        menu_option_buttons.setLayout(options_button_layout)
        k_means_options.setLayout(k_means_options_layout)

        # options_menu_layout.addWidget(box_dock_widget)
        options_menu_layout.addWidget(k_means_options)
        options_menu_layout.addWidget(menu_option_buttons)

        options.setLayout(options_menu_layout)

        #sets up the graphs for the ML methods
        self.init_graphs(graphs_layout)
        graphs.setLayout(graphs_layout)
        
        #creates the main body of the GUI
        body_layout.addWidget(graphs)
        body_layout.addWidget(options)
        
        #sets up the two larger components of the main body of the gui
        body.setLayout(body_layout)
        ribbon.setLayout(ribbon_layout)

        #sets up the main layout of the other nested widgets
        main_layout.addWidget(ribbon)
        main_layout.addWidget(body)
        
        #sets the main layout
        self.setLayout(main_layout)
        self.setCentralWidget(body)

        #adding the dock widget for the checkboxes, specifying it will start in the right most slot
        #NOTE: we have to specify that we are adding a dock widget with this method, otherwise we will not be able
        #to dock the window again once it has been detached
        self.addDockWidget(Qt.RightDockWidgetArea, box_dock_widget)


    def init_graphs(self, graphs_layout:QVBoxLayout):
        '''
            Makes the graphs for the k-means and lr machine learning algorithms

            Parameters:
                graphs_layout: the layout widget (QVBoxLayout type) that the canvases and graphs will be attached to
        '''
        xy_and_scale_default_dimension = 111

        #two figures for the different graphs
        self.k_means_figure = plt.figure(1)
        plt.title("K-Means Clustering Graph")
        self.regression_figure = plt.figure(2)
        plt.title("Linear Regression Graph")

        #gives the graphs some initial dimensions
        self.k_means_figure.add_subplot(xy_and_scale_default_dimension)
        self.regression_figure.add_subplot(xy_and_scale_default_dimension)

        #the canvases the graphs will be displayed on
        self.k_canvas = FigureCanvas(self.k_means_figure)
        self.regression_canvas = FigureCanvas(self.regression_figure)
        #the navigation tools that go to thier respective canvases
        k_toolbar = NavigationToolbar(self.k_canvas, self)
        regression_toolbar = NavigationToolbar(self.regression_canvas, self)


        #add the toolbars and widgets to the layout that was passed in
        graphs_layout.addWidget(k_toolbar)
        graphs_layout.addWidget(self.k_canvas)
        graphs_layout.addWidget(regression_toolbar)
        graphs_layout.addWidget(self.regression_canvas)

        self.k_canvas.draw()
        self.regression_canvas.draw()


    def make_checkboxes(self, index, boxes_layout:QGridLayout):
        '''
            Makes the checkboxes for the different columns for our data

            Parameters:
                index: the index of the name of the box and it's value
                boxes_layout: the GridLayout the boxes will be drawn on
        '''
        grid_size = 5
        #makes the column and row that will be used for the check box in the grid layout
        col = index % grid_size
        row = index // grid_size

        if index < len(self.COLUMN_NAMES):
            #makes a new check box and adds it to the list of references
            box = QCheckBox(self.COLUMN_NAMES[index])
            self.box_references.append(box)

            #calls the method to add and remove element depending on if the box is checked or not
            box.stateChanged.connect(lambda: self.toggle_array_element(box, self.COLUMN_VALUES[index]))

            #sets the box at it's correct column and row indecies
            boxes_layout.addWidget(box, row, col)

        else:
            #makes a button to quickly clear all the boxes and adds it to the end of the boxes layout
            clear_all_boxes_button = QPushButton("Clear All Boxes")
            clear_all_boxes_button.clicked.connect(self.uncheck_all_boxes)
            boxes_layout.addWidget(clear_all_boxes_button)
    
        
    def toggle_array_element(self, box_ref:QCheckBox, column_context):
        '''
            Method that adds and removes elements from the array of column names

            Parameters:
                box_ref: a reference to the box itself so that we can check it's own toggled state
                column_context: a string that corresponds to the column that is being toggled on or off
        '''
        
        #if the box is updated to the checked state we add the string. otherwise we remove it
        if box_ref.isChecked():
            self.column_selection.append(column_context)
        else:
            self.column_selection.remove(column_context)


    def make_menu_buttons(self, h_box_layout:QHBoxLayout):
        '''
            Makes the buttons for doing the menu options (making the dataframe, printing the columns selected, starting
            the k-means clustering algorithm, starting the linear regression algorithm)

            Parameters:
                h_box_layout: the QHBoxLayout the buttons will be drawn on
        '''

        #make the buttons and give them their labels
        self.df_button = QPushButton("Make Dataframe")
        self.print_col_button =  QPushButton("Print Columns")
        self.k_means_button =  QPushButton("K-Means Graph")
        self.linear_regression_button =  QPushButton("LR Graph")

        #set up the methods that the buttons should be connected to when they are clicked
        self.df_button.clicked.connect(self.games_controller.make_dataframe)
        self.print_col_button.clicked.connect(self.games_controller.do_selection)
        self.k_means_button.clicked.connect(lambda: self.set_menu_buttons_enabled(False))
        self.k_means_button.clicked.connect(lambda: self.set_column_confirm_button_enabled(True))
        self.linear_regression_button.clicked.connect(self.games_controller.linear_regression_filter)

        #set the buttons that cannot be used yet to be disabled
        self.set_menu_buttons_enabled(False)

        #add the buttons to the layout that was passed to the method
        h_box_layout.addWidget(self.df_button)
        h_box_layout.addWidget(self.print_col_button)
        h_box_layout.addWidget(self.k_means_button)
        h_box_layout.addWidget(self.linear_regression_button)

    
    def make_k_means_options(self, k_means_layout:QHBoxLayout):
        '''
            Creates the k-means options boxes and buttons and organizes them

            Parameters:
                k_means_layout: a Horizontal Box Layout that the buttons and boxes that are used for k-means
        '''

        #makes the confirmation button for the columns selection and the x and y axis boxes
        self.confirm_columns_button = QPushButton("Confirm Columns")
        self.confirm_columns_button.clicked.connect(lambda: self.set_checkboxes_enabled(False))
        self.confirm_columns_button.clicked.connect(lambda: self.set_column_confirm_button_enabled(False))
        self.confirm_columns_button.clicked.connect(lambda: self.set_k_means_GUI_enabled(True))
        self.confirm_columns_button.clicked.connect(self.games_controller.prep_clustering)
        self.k_means_x_axis = QComboBox()
        self.k_means_y_axis = QComboBox()

        #makes a box and sets the range to an acceptable one at this time
        self.cluster_box = QSpinBox()
        self.cluster_box.setPrefix("Number of clusters: ")
        self.cluster_box.setRange(self.MIN_CLUSTERS, self.MAX_CLUSTERS)

        self.confirm_clusters_button = QPushButton("Confirm Cluster Amount")
        self.confirm_clusters_button.clicked.connect(self.games_controller.start_k_means)

        # enable/disable GUI elements here
        self.set_k_means_GUI_enabled(False)
        self.set_column_confirm_button_enabled(False)

        k_means_layout.addWidget(self.confirm_columns_button)
        k_means_layout.addWidget(self.k_means_x_axis)
        k_means_layout.addWidget(self.k_means_y_axis)
        k_means_layout.addWidget(self.cluster_box)
        k_means_layout.addWidget(self.confirm_clusters_button)


    def uncheck_all_boxes(self):
        '''
            Unchecks all the column boxes that are checked
        '''

        #loops through all the buttons and if they're checked, uncheck them
        for box in self.box_references:
            if isinstance(box, QCheckBox) and box.isChecked():
                box.click()


    def set_checkboxes_enabled(self, enabled=True):
        '''
            Makes all the boxes and buttons for the column selections enabled or disabled

            Parameters:
                enabled: whether or not a box or button for the column selection should be enabled or disabled
        '''
        for box_or_button in self.box_references:
            box_or_button.setEnabled(enabled)


    def set_menu_buttons_enabled(self, enabled=True):
        '''
            Enable or disable the menu buttons

            Parameters:
                enabled: Bool for if the buttons should be enabled or not, defaults to True
        '''

        self.print_col_button.setEnabled(enabled)
        self.k_means_button.setEnabled(enabled)
        self.linear_regression_button.setEnabled(enabled)


    def set_column_confirm_button_enabled(self, enabled=True):
        '''
            Enables the button for the columns confirmation. Defaults to true

            Parameters:
                enabled: Bool for if the buttons should be enabled or not, defaults to True
        '''
        self.confirm_columns_button.setEnabled(enabled)
        
        #when the k-means elements are on, the others should turn off to reduce user confusion
        self.set_menu_buttons_enabled(False)


    def set_k_means_GUI_enabled(self, enabled=True):
        '''
            Sets the state of the rest of the k-means GUI to enabled or disabled. Defaults to true

            Parameters:
                enabled: Bool for if the buttons should be enabled or not, defaults to True
        '''
        self.k_means_x_axis.setEnabled(enabled)
        self.k_means_y_axis.setEnabled(enabled)
        self.cluster_box.setEnabled(enabled)
        self.confirm_clusters_button.setEnabled(enabled)


    def k_means_on_hover(self, event):
        '''
            Handles the hover event for the K-means graph

            Parameters:
                event: the mouse event that we have registered for to make the graph hoverable. Contains information
                    about the hover event that we can then get data from to make our annotations.

            Credit to: Jie Jenn on YouTube for the base example of his use of annotations that this is adapted from.
                Video Link - https://www.youtube.com/watch?v=fyGEBJBJJW0&list=PLVGYqB47-J_-bXkPy7SKkoj9KLYnwHlh8&index=1&t=897s
        '''
        min_col_len = 0

        #if you dont return when this condition is true, the command line will be flooded with text
        if self.k_means_annotations == None:
            return
        #if the mouse is within the bounds of the graph
        if event.inaxes == self.k_means_axes:
            for path_collection in self.k_means_plots:
                #this is getting if you are hovering over a point
                is_contained, annotation_index = path_collection.contains(event)
                if is_contained:
                    break

            if is_contained:
                #getting the location of the data point and telling the annotation which x and y coordinates it should use
                #NOTE:this uses closure on the path_collection variable
                data_point_location = path_collection.get_offsets()[annotation_index['ind'][0]]
                self.k_means_annotations.xy = data_point_location
                #NOTE: keep in mind that this game array is actually a 2d array.
                # 1st dimension is the entire game, 2nd dimension are the values of the game that we make when the dictionary is formed
                game_array_2d = self.adjust_key_for_price_scaling(data_point_location)

                #make and set the label for the annotation and make it visible and draw it
                if len(self.column_selection) > min_col_len:
                    appID = game_array_2d[self.FIRST_GAME_IDX][self.APPID]
                    #form for this is Row(col-0=col_info-0, col-1=col_info-1,..., col-n=col_info-n)
                    hover_info_row = self.games_controller.get_hover_info(appID).select(self.column_selection).collect()
                    #constructing the string using the info the user has selected from the columns
                    text_label = ""
                    #if the "*" option is selected then we need to adjust how the string is made
                    if "*" in self.column_selection:
                        for i in range(len(hover_info_row[0])):
                            text_label += str(self.COLUMN_NAMES[i]) + ": " + \
                                                                    str(hover_info_row[self.FIRST_GAME_IDX][i])+ "\n"
                    #if the "*" option is NOT selected then the string does not need to be constructed differently
                    else:
                        for i in range(len(hover_info_row[0])):
                            text_label += str(self.column_selection[i]) + ": " + \
                                                                    str(hover_info_row[self.FIRST_GAME_IDX][i]) + "\n"

                else:
                    text_label = "AppID: {0}\nName: {1}".format(game_array_2d[self.FIRST_GAME_IDX][self.APPID],
                                                                  game_array_2d[self.FIRST_GAME_IDX][self.APPNAME])
                
                self.k_means_annotations.set_text(text_label)
                self.k_means_annotations.set_visible(True)
                self.k_means_figure.canvas.draw_idle()
            else:
                #stop drawing the annotations if you are not hovering over a point
                self.k_means_annotations.set_visible(False)
                self.k_means_figure.canvas.draw_idle()


    def regression_on_hover(self, event):
        '''
            Handles the hover event for the K-means graph

            Parameters:
                event: the mouse event that we have registered for to make the graph hoverable. Contains information
                    about the hover event that we can then get data from to make our annotations.

            Credit to: Jie Jenn on YouTube for the base example of his use of annotations that this is adapted from.
                Video Link - https://www.youtube.com/watch?v=fyGEBJBJJW0&list=PLVGYqB47-J_-bXkPy7SKkoj9KLYnwHlh8&index=1&t=897s
        '''


        #if you dont return when this condition is true, the command line will be flooded with text
        if self.regression_annotations == None:
            return
        if event.inaxes == self.regression_axes:
            #this is getting if you are hovering over a point
            is_contained, annotation_index = self.regression_plot.contains(event)
            if is_contained:
                #getting the location of the data point and telling the annotation which x and y coordinates it should use
                data_point_location = self.regression_plot.get_offsets()[annotation_index['ind'][self.DATAPOINT_IDX]]
                self.regression_annotations.xy = data_point_location

                #make and set the label for the annotation and make it visible and draw it
                text_label = "Year: {0}\nGames Published to Steam: {1}".format(data_point_location[self.APPID], \
                                                                                 data_point_location[self.APPNAME])
                self.regression_annotations.set_text(text_label)
                self.regression_annotations.set_visible(True)
                self.regression_figure.canvas.draw_idle()
            else:
                #stop drawing the annotations if you are not hovering over a point
                self.regression_annotations.set_visible(False)
                self.regression_figure.canvas.draw_idle()


    def adjust_key_for_price_scaling(self, data_point_location):
        '''
            Takes the datapoint location from k-means and adjusts the value upward by a factor of 100 if the
            axis is for price

            Parameters:
                data_point_location: the datapoint location that triggered the hover event on k-means

            Return:
                The 2D array for the hover entry on the k-means graph
        '''

        upscale = 100
        precission = 1
        x_dict_dimension = 0
        y_dict_dimension = 1

        dimension1 = str(data_point_location[x_dict_dimension])
        dimension2 = str(data_point_location[y_dict_dimension])
        # if the first dimenstion key needs to be adjusted upwards for the price
        if dimension1 not in self.k_means_annotations_dict:
            #this seems to be the solution for adjusting the price, needs to round and have 1 point of precission
            dimension1 = str(round(data_point_location[x_dict_dimension] * upscale, precission))
        # if the second dimenstion key needs to be adjusted upwards for the price
        if dimension2 not in self.k_means_annotations_dict[dimension1]:
            #this seems to be the solution for adjusting the price, needs to round and have 1 point of precission
            dimension2 = str(round(data_point_location[y_dict_dimension] * upscale, precission))
        #returns the 2d array that should be at the key location of the first and second layer of the nested dictionaries
        return self.k_means_annotations_dict[dimension1][dimension2]


if __name__ == "__main__":
    '''
        The point of first contact with the program. Checks command line arguments and makes the main window for the
        GUI.
    '''
    # creating a pyqt5 application and checking if there are enough command line arguments for the program
    app = QApplication(sys.argv)
    if len(sys.argv) != MIN_ARGS:
            print(USAGE)
            exit(1)

    #makes a MainMenu and changes the title of the window
    main_program = MainMenu()
    main_program.setWindowTitle("Steam Games App")
    
    #resets the window to be at the front of the screen and active since it defaults to being in the back
    main_program.setFocus()
    main_program.activateWindow()
    main_program.setWindowState(Qt.WindowState())
    main_program.show()
  
    #wait until the user clicks to exit the program to close
    sys.exit(app.exec_())