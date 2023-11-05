# Data Fitting and Analysis Tool

	This tool performs data fitting and plotting operations using a linear regression model on time-series data provided in a CSV file.

	The script will check the time intervals, interpolate faults, generate daily patterns, fit a regression model, and provide outputs: a PNG file with the plotted results, console output of the model's fit metrics, including MSE percentage and explained variability percentage.



## Requirements

	You need Python 3 to run this script with the following modules installed:

	- pandas
	- numpy
	- scikit-learn
	- matplotlib


	
### Usage

	Run the script from the command line by passing the input CSV file and the name of the quantity column as arguments:

		python script.py --input_file "path_to_csv.csv" --quantity "column_name"

		example:
			python "C:\Users\marek\Desktop\DA\ukol_Watt\Watt_v1\script.py" --input "C:\\Users\\marek\\Desktop\\DA\\ukol_Watt\\SG.csv" --quantity "Consumption"


	Input File Format:

		The input CSV file must have a datetime column followed by data columns.

	
	Output File Format
		
		The output PNG file is saved in the current working directory with the name:
		
			input_file '_' quantity '_D' date of computation '_T' time of computation '.png'