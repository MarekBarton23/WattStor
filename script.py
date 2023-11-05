import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import os


def main(input_file, quantity):
    """
    Execute main routine processing a dataset with fitting and plotting operations.

    This function reads data from a CSV file, checks for correct time point intervals,
    interpolates faults, generates daily patterns, fits a model, and saves a plot to file.

    Requires a CSV file with specific structure: a datetime column followed by
    data columns.

    Outputs plot to PNG file and prints fit metrics to the console.
    """
    def check_timepoints(dframe):
        """
        Verify sequential timing in dataset with 30-minute intervals.

        Raises:
            ValueError: If data points are not equidistant by 30 minutes.

        Args:
            dframe (pd.DataFrame): DataFrame with a datetime column as the first column.

        Returns:
            None
        """
        timeline = pd.to_datetime(dframe.iloc[:, 0])
        inconsistencies = timeline[timeline.diff() != pd.Timedelta(minutes=30)].index
        inconsistencies = inconsistencies[1:]
        inconsistent_values = df.iloc[inconsistencies].values
        if len(inconsistent_values) > 0:
            raise ValueError("The data are not sampled equidistantly!")

    def daily_pattern(column_data):
        """
        Average values across days to generate a daily pattern for a dataset variable.

        Args:
            column_data (np.ndarray): 1D array of numerical values representing a variable over time.

        Returns:
            np.ndarray: 1D array of averaged values representing the daily pattern.
        """
        check_format(column_data)
        nsegments = len(column_data) // 48
        segments = column_data[:nsegments * 48]
        matrix = segments.reshape(-1, 48)
        valid_indices = ~np.isnan(matrix).any(axis=1)
        matrix = matrix[valid_indices]
        return np.mean(matrix, axis=0)

    def correct_faults(data):
        """
        Interpolate missing or infinite values in a dataset using linear interpolation.

        Args:
            data (np.ndarray): 1D array of numbers with possible NaN or infinite values.

        Returns:
            np.ndarray: 1D array of numbers with missing values interpolated.
        """
        check_format(data)
        invalid_mask = np.isnan(data) | np.isinf(data)
        valid_points = np.where(~invalid_mask)[0]
        data[invalid_mask] = np.interp(np.flatnonzero(invalid_mask), valid_points, data[valid_points])
        return data

    def fitting(data, timecourse):
        """
        Model and fit data to a representative time course using linear regression.

        Raises:
            ValueError: If either input contains no variance.

        Args:
            data (np.ndarray): 1D array of the variable to be modeled.
            timecourse (np.ndarray): 1D array representing the typical daily pattern.

        Returns:
            np.ndarray: Predictions from the fitted model.
        """
        check_format(data)
        check_format(timecourse)
        if not timecourse.std() or not data.std():
            raise ValueError("Both inputs needs to contain variance.")

        M, N = len(data), len(timecourse)
        X = np.zeros(((int(M // N)+1)*N, int(M // N) * 2+2))
        for i in range(0, M, N):
            X[i:i + N, int(i / N) * 2] = timecourse
            X[i:i + N, int(i / N) * 2 + 1] = 1
        X = X[0:M, :]
        model = LinearRegression()
        model.fit(X, data)
        return model.predict(X)

    def plot_to_file(input_file, quantity, data, modelled):
        """
        Save plot of data and modelled fit to a file and print fit metrics.

        Args:
            input_file (str): Path to the CSV file used as the data source.
            quantity (str): The measured variable's column name in the CSV file.
            data (np.ndarray): 1D array of observed values.
            modelled (np.ndarray): 1D array of values predicted by the model.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        plt.plot(data, label='Data')
        plt.plot(modelled, 'r--', label='Data Fit')
        mse_perc = (np.mean((data - modelled) ** 2) / np.mean(data ** 2)) * 100
        expl_var = (1 - np.var(data - modelled) / np.var(data)) * 100

        plt.legend(loc='upper left')
        metrics_text = (f"MSE [%]: {mse_perc:.2f}\n"
                        f"Explained Variability [%]: {expl_var:.2f}")

        plt.figtext(0.5, -0.1, metrics_text, ha="center", fontsize=10,
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

        base_file_name = os.path.splitext(os.path.basename(input_file))[0].replace('.', '_')
        date_time = datetime.now().strftime("D_%Y%m%d_T_%H%M%S")
        file_name = f"{base_file_name}_{quantity}_{date_time}.png"

        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
        print(metrics_text)

    def check_format(data):
        """
        Check if input is a 1D array of numbers and has at least two samples.

        Raises:
            ValueError: If input is not a 1D array of numbers or has fewer than two samples.

        Args:
            data (np.ndarray): The input data array to check.

        Returns:
            None
        """
        if not isinstance(data, np.ndarray) or data.ndim != 1 or not np.issubdtype(data.dtype, np.number):
            raise ValueError("The input must be a 1D numpy array containing numbers.")
        if len(data) < 2:
            raise ValueError("The input must have at least two samples.")

    df = pd.read_csv(input_file, delimiter=';')
    col_data = df[quantity].to_numpy()
    check_timepoints(df)
    average_profile = daily_pattern(col_data)
    modelled = fitting(correct_faults(col_data), average_profile)
    plot_to_file(input_file, quantity, col_data, modelled)


if __name__ == "__main__":
    """The main function callable from terminal with parameters --.csv file and ---quantity"""
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--quantity', type=str, required=True, help='Quantity to process')
    args = parser.parse_args()

    main(args.input, args.quantity)
