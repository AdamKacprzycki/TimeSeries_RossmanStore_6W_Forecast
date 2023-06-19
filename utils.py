import sys
from typing import List

import inflection

import pandas as pd
import numpy as np
import math

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import periodogram

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

###############################################-DTYPE OPT_MEMORY USAGE OPT-###############################################

def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    
    '''
    The function optimize_floats takes a pandas DataFrame as input and optimizes the memory usage of float columns in the DataFrame. 
    It first selects all float columns using select_dtypes method with include=['float64'] parameter, and converts them to the smallest 
    possible float type using pd.to_numeric method with downcast='float' parameter. Finally, the function returns the modified DataFrame.
    '''
    
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    
    '''
    The function optimize_ints takes a pandas DataFrame as input and optimizes the memory usage of integer columns in the DataFrame. 
    It first selects all integer columns using select_dtypes method with include=['int64'] parameter, and converts them to the smallest 
    possible integer type using pd.to_numeric method with downcast='integer' parameter. Finally, the function returns the modified DataFrame. 
    '''
        
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def optimize_objects(df: pd.DataFrame, datetime_features: List[str], bool_features: List[str]) -> pd.DataFrame:
    '''
    The function optimize_objects takes a pandas DataFrame as input, along with a list of column names (datetime_features) that are supposed to be 
    of datetime type, and a list of column names (bool_features) that are supposed to be of bool type. The function optimizes the memory usage of 
    object columns in the DataFrame.

    It first loops over all object columns in the DataFrame using select_dtypes method with include=['object']. For each column, if it is not in 
    datetime_features or bool_features and not of type list, the function checks if the ratio of the number of unique values to the total number 
    of values in the column is less than 0.5. If it is, the column is converted to the 'category' data type, which can save memory compared to 
    object data type.

    If a column is in datetime_features, it is converted to datetime type using pd.to_datetime method.
    
    If a column is in bool_features, it is converted to bool type using astype(bool) method.
    
    Finally, the function returns the modified DataFrame.
    '''
    
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features and col not in bool_features:
            if not (type(df[col][0]) == list):
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if float(num_unique_values) / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        elif col in datetime_features:
            df[col] = pd.to_datetime(df[col])
    
    for col in bool_features:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    return df


def optimize(df: pd.DataFrame, datetime_features: List[str] = [], bool_features: List[str] = []):
    '''
    The function optimize takes a pandas DataFrame as input, along with an optional list of column names (datetime_features) that are supposed to be of
    datetime type, and an optional list of column names (bool_features) that are supposed to be of bool type. The function optimizes the memory usage of 
    integer, float, and object columns in the DataFrame using the optimize_objects, optimize_ints, and optimize_floats functions.

    The function first calculates the memory usage of the input DataFrame using sys.getsizeof method. It then calls the three optimization functions in the
    following order: optimize_objects, optimize_ints, and optimize_floats.

    After the optimization is complete, the function prints out the percentage of memory reduction achieved by the optimization. 
    Finally, the function does not return anything, nevertheless, it changes the types of the passed DataFrame.
    '''
    
    mem_usage_bef = sys.getsizeof(df)
    df = optimize_objects(df, datetime_features, bool_features)
    optimize_ints(df)
    optimize_floats(df)
    print(f'Optimize_memory_func reduce memory usage by {(1 - (sys.getsizeof(df) / mem_usage_bef)) * 100 :.2f} %.')
    
###############################################-OUTLIER-DETECTION-BOXPLOTS-###############################################
    
def turkey_horizontal_boxplot(dataframe: pd.DataFrame, 
                       x_feature: str, 
                       y_feature:str = None):
    """
    Description: 
    
    This function creates a horizontal boxplot using Seaborn library. It takes in a pandas dataframe and one or two feature/column names as input, 
    and generates a horizontal boxplot to visualize the distribution of data.
    
    Args:

    dataframe (pandas dataframe): The input dataframe containing the data to be plotted.
    x_feature (str): The name of the column in the dataframe to be plotted on the x-axis.
    y_feature (Optional[str]): The name of the column in the dataframe to be plotted on the y-axis. If this parameter is not provided, only the x_feature will be plotted.
    
    Return:

    sns.boxplot
    
    Dependencies:
    
    This function requires the pandas and numpy libraries to be installed.
    """
    
    # Set the theme and style for the plot using Seaborn
    sns.set_theme(rc={'figure.figsize': (10,6)},style='whitegrid',palette='Reds')
    
    # If y_feature is provided, plot a boxplot with both x_feature and y_feature
    if y_feature != None:
        ax = sns.boxplot(x=dataframe[x_feature], y=dataframe[y_feature])
        ax.set_title(f'Boxplot \n x: {x_feature} -  y: {y_feature}') # Set the plot title to display feature names

    # If y_feature is not provided, plot a boxplot with only x_feature
    else:
        ax = sns.boxplot(x=dataframe[x_feature], y=None)
        ax.set_title(f'Boxplot \n x: {x_feature}')

        
def tukey_method_outliers(
    dataframe: pd.DataFrame, 
    column: str) -> tuple[float, float]:
    
    """
    Description:
    This function applies the Tukey's method to identify potential outliers in a column of a given dataframe.
    The method calculates the upper and lower bounds for potential outliers based on the interquartile range (IQR)
    of the distribution, as 1.5 times the IQR above and below the 75th and 25th percentiles, respectively.
    
    Args:
    - dataframe: pandas.DataFrame object, the dataframe containing the column of interest
    - column: str, the name of the column for which to identify outliers
    
    Return:
    A tuple containing the upper and lower bounds for potential outliers.
    
    Dependencies:
    This function requires the pandas and numpy libraries to be installed.
    
    """
    percentile_25 = np.nanpercentile(dataframe[column], 25)
    percentile_75 = np.nanpercentile(dataframe[column], 75)
    iqr = (percentile_75 - percentile_25)
    upper_outlier_bound = percentile_75 + 1.5*iqr
    lower_outlier_bound = percentile_25 - 1.5*iqr

    return (upper_outlier_bound, lower_outlier_bound)

def tukey_method_outlier_count_and_percentage(
    dataframe: pd.DataFrame, 
    column: str) -> tuple[int, float]:
    
    """
    Description:
    This function uses the Tukey's method to identify potential outliers in a column of a given dataframe,
    and calculates the count and percentage of potential outliers in the column.

    Args:
    - dataframe: pandas.DataFrame object, the dataframe containing the column of interest
    - column: str, the name of the column for which to identify outliers

    Return:
    A tuple containing the count and percentage of potential outliers in the column.

    Dependencies:
    This function requires the pandas and numpy libraries to be installed.
    """
    upper_outlier_bound, lower_outlier_bound = tukey_method_outliers(dataframe, column)

    count = 0
    for value in dataframe[column]:
        if value > upper_outlier_bound or value < lower_outlier_bound:
            count += 1
    percentage = round(count / dataframe.shape[0] * 100, 2)

    return (count, percentage)
    
###############################################-OUTLIER-DETECTION-FUNCTION-###############################################

def custom_outlier_catcher_2d_sales_customers_and_agg(
    dataframe: pd.DataFrame,
    feature: str,
    threshold_max_coeff=0.9,
    threshold_max: float = 99.998,
    threshold_min: float = 99.9899,
    show_outliers: bool = True,
    remove_outliers: bool = False,
    df_output: bool = False,
) -> pd.DataFrame:
    
    """
    Description:
    
    This function detects and visualizes outliers in a 2D sales and customers dataset. It calculates 
    a threshold based on the correlation coefficient between the two features and a percentile value. 
    It then detects outliers based on the residuals from a linear regression model fit on the dataset. 
    It removes the outliers and plots the data points and regression line for each unique value of the 
    feature column specified. 
    
    
    Potential outliers are identified by a highlighted point, while the point farthest from the regression 
    line is marked with a star.
    
    Args:
    - dataframe: A pandas DataFrame object containing the 2D sales and customers dataset.
    - feature: A string indicating the name of the column containing the unique values to group the data.
    - threshold_max_coeff: A float indicating the maximum correlation coefficient threshold to use for 
      calculating the outlier detection threshold. Defaults to 0.9.
    - threshold_max: A float indicating the maximum percentile value to use for calculating the outlier 
      detection threshold. Defaults to 99.998.
    - threshold_min: A float indicating the minimum percentile value to use for calculating the outlier 
      detection threshold. Defaults to 99.9899.
    - show_outliers: A boolean indicating whether or not to show the outliers in the plot. Defaults to True.
    - remove_outliers: A boolean indicating whether or not to remove the outliers from the dataset. Defaults to False.
    - df_output: A boolean indicating whether or not to return a copy of the DataFrame without the outliers. 
      Defaults to False.
    
    Return:
    - A pandas DataFrame object (copy of initial df) with the outliers removed. Only returned if df_output is set to True.
    
    Dependencies:
    - pandas, numpy, math, matplotlib, seaborn
    """
    
    unique_values = sorted(dataframe[feature].unique().tolist())

    columns_num = 2
    rows_num = math.ceil(len(unique_values) / columns_num)
    df_no_outliers = dataframe.copy()

    fig, axes = plt.subplots(rows_num, columns_num, figsize=(10 * columns_num, 8 * rows_num))

    row = -1
    column = columns_num - 1

    for unique_value in unique_values:
        if column == (columns_num - 1):
            row += 1
            column = 0
        else:
            column += 1
        temp_df = df_no_outliers[df_no_outliers[feature] == unique_value]

        # Outlier detection
        x = temp_df['Customers'].values
        y = temp_df['Sales'].values
        
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        residuals = y - model.predict(x.reshape(-1, 1))
        max_residual_idx = np.argmax(np.abs(residuals))
        max_residual = residuals[max_residual_idx]

        # Threshold calculation
        corr_coef = temp_df['Customers'].corr(temp_df['Sales'])

        if corr_coef > threshold_max_coeff:
            threshold = np.percentile(np.abs(residuals), threshold_max)
        else:
            threshold = np.percentile(np.abs(residuals), threshold_min)

        outliers = np.where(np.abs(residuals) > threshold)[0]

        # Remove outliers from df_no_outliers
        if remove_outliers:
            df_no_outliers = df_no_outliers.drop(temp_df.index[outliers])

        # Plot data and regression line
        sns.scatterplot(ax=axes[row, column], x='Customers', y='Sales', color='lightcoral', alpha=0.7, data=temp_df)
        lin_fit = np.polyfit(temp_df['Customers'], temp_df['Sales'], 1)
        lin_func = np.poly1d(lin_fit)(temp_df['Customers'])
        axes[row, column].plot(temp_df['Customers'], lin_func, "k--", lw=1)

        if show_outliers:
            axes[row, column].scatter(x[max_residual_idx], y[max_residual_idx], color='red', marker='*', s=200)
            axes[row, column].scatter(x[outliers], y[outliers], color='indianred', marker='o', s=100)

        axes[row, column].set_title(f"Sales vs Customers for {feature} - {unique_value}\nCorrelation = {round(temp_df['Customers'].corr(temp_df['Sales']) * 100, 2)}%")

    # Remove any excess axes
    if len(unique_values) % columns_num != 0:
        for column_num in range(column + 1, columns_num):
            fig.delaxes(axes[rows_num - 1][column_num])

    if df_output:
        print(f'The identified outliers have been successfully removed, resulting in a reduction of {dataframe.shape[0] - df_no_outliers.shape[0]} data points in the initial DataFrame.')
        
        return df_no_outliers
    
    
###############################################-SEASONAL_DECOMPOSE-FUNCTION-###############################################

def custom_seasonal_decompose(
    df: pd.DataFrame,
    column: str = 'Sales',
    title: str = 'Decomposition of ...',
    resample: str ='D',
    col_date: str = 'Date', 
    color: str = 'firebrick',
    set_date_index: bool = True
) -> None:
    
    """
    Perform seasonal decomposition of a time series and display the components.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        column (str, optional): The name of the column to analyze.
        col_date (str, optional): Name of the date column in the DataFrame. Defaults to 'Date'.
        color (str, optional): Color for the plots. Defaults to 'firebrick'.

    Returns:
        None (display the seasonal_decompose.plot())

    """
    
    if set_date_index:

        # Convert the date column to datetime format
        df[col_date] = pd.to_datetime(df[col_date])

        # Set the date column as the DataFrame index
        df = df.set_index(col_date)

    # Aggregate data to daily level (e.g., calculate mean)
    df_daily = df[column].resample(resample).mean().dropna()

    # Perform seasonal decomposition of the time series
    decomposition = sm.tsa.seasonal_decompose(df_daily, model='additive', period=7)

    # Display the plots
    fig, axes = plt.subplots(4, 1, figsize=(18, 12))

    # Plot the Observed component
    axes[0].plot(df_daily, color=color)
    axes[0].set_ylabel('Observed')

    # Plot the Trend component
    axes[1].plot(decomposition.trend, color=color)
    axes[1].set_ylabel('Trend')

    # Plot the Seasonal component
    axes[2].plot(decomposition.seasonal, color=color)
    axes[2].set_ylabel('Seasonal')

    # Plot the Residual component
    axes[3].scatter(decomposition.resid.index, decomposition.resid, color=color, marker='.')
    axes[3].axhline(y=0, color='black', linestyle='-')
    axes[3].set_ylabel('Residual')

    # Set the title for the entire plot
    fig.suptitle(title)

    # Display the plot
    plt.show()


def custom_plot_acf(
    df: pd.DataFrame, 
    column: str,
    date_col: str = 'Date',
    color: str = 'firebrick', lags: int = 30,
    set_date_index: bool = True,
    resample_df: bool = True
) -> None:
    
    """
    Plot the autocorrelation function (ACF) for a given column in a DataFrame.

    Parameters:
        - df (DataFrame): The input DataFrame.
        - column (str): The name of the column to analyze.
        - date_col (str): The name of the column containing the dates. Default is 'Date'.
        - color (str): The color of the ACF plot. Default is 'firebrick'.
        - lags (int): The number of lags to display in the ACF plot. Default is 30.

    Returns:
        None (displays the ACF plot).
    """

    if set_date_index:
            
    # Set the date column as the index
        df = df.set_index('Date')
        
    if resample_df:
        
    # Resample the column to daily frequency and calculate the mean
        df_daily = df[column].resample('D').mean().dropna()
    else:
        df_daily = df

    # Configure plot settings
    plt.rc("figure", autolayout=True, figsize=(11, 3))
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[color])

    # Plot the ACF
    plot_acf(df_daily, lags=lags)
    plt.xticks(np.arange(1,lags+1,1))

    # Show the plot
    plt.show()
    

def custom_plot_pacf(
    df: pd.DataFrame, 
    column: str,
    date_col: str = 'Date',
    color: str = 'firebrick',
    lags: int = 30,
    set_date_index: bool = True,
    resample_df: bool = True
    
) -> None:
    """
    Plot the partial autocorrelation function (PACF) for a given column in a DataFrame.

    Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - column (str): The name of the column to analyze.
        - date_col (str): The name of the column containing the dates. Default is 'Date'.
        - color (str): The color of the PACF plot. Default is 'firebrick'.
        - lags (int): The number of lags to display in the PACF plot. Default is 30.

    Returns:
        None (displays the PACF plot).
    """

    if set_date_index:
        
    # Set the date column as the index
        df = df.set_index(date_col)

    if resample_df:
    # Resample the column to daily frequency and calculate the mean
        df_daily = df[column].resample('D').mean().dropna()
    else:
        df_daily = df

    # Configure plot settings
    plt.rc("figure", autolayout=True, figsize=(11, 3))
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[color])

    # Plot the PACF
    plot_pacf(df_daily, lags=lags)
    plt.xticks(np.arange(1,lags+1,1))

    # Show the plot
    plt.show()

    
def rolling_mean_analysis(
    df: pd.DataFrame, 
    window_sizes: list = [7, 30, 365], 
    date_column: str = 'Date', 
    column: str = 'Sales',
    set_date_index: bool = True,
    resample_df: bool = True
    
)-> None:
    """
    Analyze the stationarity of a time series by plotting rolling statistics.

    Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - window_sizes (list): A list of window sizes for calculating rolling statistics.
        - date_column (str): The name of the date column. Default is 'Date'.
        - column (str): The name of the column to analyze. Default is 'Sales'.

    Returns:
        None (displays the plot).
    """
    
    if set_date_index:
        
    # Set the date column as the index
        df = df.set_index(date_column)
        
    if resample_df:
    # Resample the column to daily frequency and calculate the mean
        df_daily = df[column].resample('D').mean().dropna()
    else:
        df_daily = df
        
    fig, ax = plt.subplots(len(window_sizes), 1, figsize=(16, 10))

    for i, window in enumerate(window_sizes):
        rolling_mean = df_daily.rolling(window=window, center=True, min_periods=math.ceil(window/2)).mean()
        rolling_std = df_daily.rolling(window=window, center=True, min_periods=math.ceil(window/2)).std()

        ax[i].plot(df_daily, label='Original', color='firebrick', alpha=0.5)
        ax[i].plot(rolling_mean, label='Moving Average', color='firebrick', alpha=1)
        ax[i].plot(rolling_std, label='Moving Std', color='dimgray', alpha=0.7)
        ax[i].set_title(f'{window}-Day Window')
        ax[i].legend()

    plt.tight_layout()
    plt.show()
    
#####################################-STATIONARITY-TEST-UNIT-ROOT-FUNCTION-###############################################
    
def custom_test_stationarity(
    df: pd.DataFrame, 
    col_date: str = 'Date', 
    column: str = 'Sales', 
    resample: str ='D', 
    full_results: bool =True,
    make_diff: bool = False,
    num_diff: int = 1,
    set_date_index: bool = True,
    return_diffed_df = False
):
    
    """
    Description:
    
    Perform stationarity tests on a time series using the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.
    
    To ensure the stationarity of the series, it is recommended to apply both tests.
    
    Possible outcomes of applying these stationary tests are as follows:

    Case 1: Both tests conclude that the series is not stationary - The series is not stationary
    Case 2: Both tests conclude that the series is stationary - The series is stationary
    Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. 
            Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
    Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. 
            Differencing is to be used to make series stationary. The differenced series is checked for stationarity.

     Parameters:
        - df (pd.DataFrame): DataFrame containing the time series.
        - col_date (str): Name of the column containing dates (time series index).
        - column (str): Name of the column containing the time series values.
        - resample (str): Time unit to resample the time series to (default is 'D' for day).
        - full_results (bool): Specifies whether to display full test results (default is True).
        - make_diff (bool): Indicates whether to difference the time series (default is False).
        - no_diff (int): Number of times to difference the time series (default is 1).
    
    Based on:
    - https://www.statsmodels.org/devel/examples/notebooks/generated/stationarity_detrending_adf_kpss.html

    Returns:
    None - Displays the results of the stationarity tests.
    """
    
    if set_date_index:
        
    # Set the DataFrame index to the specified column
        df = df.set_index(col_date)
    
    # Resample the time series to the specified frequency
    df_daily = df[column].resample(resample).mean().dropna()
    
    # Apply differencing if make_diff is True
    if make_diff:
        df_daily = df_daily.diff(num_diff).dropna()
    
    # ADF test
    adf_result = adfuller(df_daily, autolag="AIC")
    adf_p_val = adf_result[1]
    adf_output = pd.Series(adf_result[0:4], index=[
        "Test Statistic",
        "p-value",
        "#Lags Used",
        "Number of Observations Used"
    ])
    for key, value in adf_result[4].items():
        adf_output["Critical Value (%s)" % key] = value
    
    # KPSS test
    kpss_result = kpss(df_daily, regression="c", nlags="auto")
    kpss_p_val = kpss_result[1]
    kpss_output = pd.Series(kpss_result[0:3], index=[
        "Test Statistic",
        "p-value",
        "Lags Used"
    ])
    for key, value in kpss_result[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    
    # Print the results
    print("Results of Stationarity Tests: \n")
    
    if full_results:
        print("- ADF Test:")
        print(adf_output,"\n")
        print("- KPSS Test:")
        print(kpss_output,'\n')
    else:
        print("ADF p-value:  {:.10f}".format(adf_output["p-value"]))
        print("KPSS p-value: {:.10f}".format(kpss_output["p-value"]),'\n')
   
    cases = {
        (True, False): "Case 2: Both tests conclude that the series is stationary - The series is stationary",
        (False, True): "Case 1: Both tests conclude that the series is not stationary - The series is not stationary",
        (False, False): "Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.",
        (True, True): "Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity."
    }

    print(cases[(adf_p_val <= 0.05, kpss_p_val <= 0.05)])
    
    if return_diffed_df:
        return df_daily
    else:
        pass
    
###############################################-CORR-HEATMAP-###############################################
    
def plot_correlation_heatmap(
    df:pd.DataFrame
) -> None:
    
    """
    Description:
    
    The plot_correlation_heatmap function takes a pandas DataFrame df as input and generates a heatmap to visualize the correlation matrix of the features in the DataFrame.
    
    Parameters:
    - df (pandas DataFrame): The input DataFrame containing the features.
    
    Return:
    - this function does not return any values. It generates a correlation heatmap plot.
    
    """
    
    upper_triangle = np.zeros_like(df.corr(), dtype = np.bool)
    upper_triangle[np.triu_indices_from(upper_triangle)] = True
    f, ax = plt.subplots(figsize = (15, 10))
    sns.heatmap(df.corr(),ax=ax,mask=upper_triangle,annot=True, fmt='.2f',linewidths=0.5,cmap='Reds');
    
###############################################-SEASONALITY-CHECK-###############################################

def custom_plot_periodogram(
    df: pd.DataFrame, 
    detrend: str = 'linear', 
    ax = None, 
    set_date_index: bool = False, 
    resample_df: bool = False,
    fig_size: tuple = (15,8)
):
    
    """
    Description:
    
    Generate a periodogram plot to visualize the frequency components of a time series.

    Args:
        df (pd.DataFrame): The input time series data.
        detrend (str, optional): The detrend method to apply. Defaults to 'linear'.
        ax (Axes, optional): The target axes to plot the periodogram. If None, a new figure and axes will be created. Defaults to None.
        set_date_index (bool, optional): Flag indicating whether to set the date column as the index. Defaults to False.
        resample_df (bool, optional): Flag indicating whether to resample the data to daily frequency. Defaults to False.

    Returns:
        Axes: The plotted periodogram.

    """
    
    if set_date_index:
        # Set the date column as the index
        df = df.set_index('Date')
    
    if resample_df:
        # Resample the column to daily frequency and calculate the mean
        df_daily = df.resample('D').mean().dropna()
    else:
        df_daily = df
    
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    frequencies, spectrum = periodogram(df_daily, fs=fs, detrend=detrend, window="boxcar", scaling='spectrum')
    
    if ax is None:
        _, ax = plt.subplots(figsize=fig_size)
    
    ax.step(frequencies, spectrum, color="firebrick")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels([
        "Annual (1)",
        "Semiannual (2)",
        "Quarterly (4)",
        "Bimonthly (6)",
        "Monthly (12)",
        "Biweekly (26)",
        "Weekly (52)",
        "Semiweekly (104)"
    ], rotation=30)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    
    return ax

##############################################-DATA-PREP--######################################################

def X_train_test_info(X, X_train, X_test):
    """
    Displays information about the data split into training and test sets.

    Parameters:
    - X (pandas.DataFrame): The original data.
    - X_train (pandas.DataFrame): The training set.
    - X_test (pandas.DataFrame): The test set.

    Returns:
    This function does not return any value; it only prints information to the standard output.

    Description:
    This function takes three arguments: X, X_train, and X_test, which are pandas.DataFrame objects. It displays information about the data split into training and test sets.
    The function first prints the results related to the data split, such as the initial size of X and the percentage size of X_train and X_test relative to X. It then displays the combined size of X_train and X_test.
    Next, the function prints the date index ranges in X_train and X_test, along with their minimum and maximum values.

    Example usage:
    X_train_test_info(X_data, X_train_data, X_test_data)
    """
    print(f'Checking split results: \
          \n Initial X size: {X.shape[0]},\
          \n X_train split size [%]: {np.round(X_train.shape[0] / X.shape[0],2) * 100},\
          \n X_test split size [%]: {np.round(X_test.shape[0] / X.shape[0],2) * 100},\
          \n X_train + X_test size: {X_train.shape[0] + X_test.shape[0]}\n')

    print(f'X_train date_index [min, max] {(X_train.index.min(), X_train.index.max())}')
    print(f'X_test date_index [min, max] {(X_test.index.min(), X_test.index.max())}')


