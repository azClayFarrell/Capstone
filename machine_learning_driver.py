import sys
import src.regression as regression

#The usage message for this program
USAGE = "\n=========================================================================================\n" \
        "Usage: [path to spark-submit script] [path to machine_learning_driver.py]\n" \
        "=========================================================================================\n"
#The minimum number of command line args
MAX_ARGS = 2

def main():
    if len(sys.argv) > MAX_ARGS:
        print(USAGE)
        exit(1)
    my_regression = regression.Regression_Algo()
    my_regression.main_menu()


if __name__ == "__main__":
    main()