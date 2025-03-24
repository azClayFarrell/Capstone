To start the program you must enter the path to the spark submit script, the path to the main file, and the path to the game data CSV file as command line arguments:

    [Path to spark-submit][Path to Main Program file][Path to Steam Data CSV]

An example of a correct command line entry follows assuming you are using a windows machine:

    %SPARK_HOME%\bin\spark-submit D:\Capstone\CapstoneFiles\main_steam_games_GUI.py D:\Capstone\SteamData\steam_games\games.csv
