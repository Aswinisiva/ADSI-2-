# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:31:18 2023

@author: Aswini
"""

import pandas as pd #Importing pandas for analysis
import matplotlib.pyplot as plt #Importing matplotlib for Data Visualisation
import seaborn as sns  #Import the seaborn for statistical data visualization
from scipy.stats import skew, kurtosis # Importing functions from scipy.stats

def read_and_process_data(file_path, indicator, countries):
    ''' Define a function to read and process data from a CSV file
        Arguments:
      #   - file_path: The path to the CSV file containing the data
      #   - indicator: The specific indicator to filter the data
      #   - countries: A list of countries to filter the data'''
    
    # Read the CSV file into a pandas DataFrame, skipping the first 4 rows
    df = pd.read_csv(file_path, skiprows=4)

    # Filter the DataFrame based on the specified indicator and countries
    data = df[(df["Indicator Name"] == indicator) & (df["Country Name"].
                                                isin(countries))]
      
    # Select the columns needed for analysis
    columns_needed = ['Country Name', '2004','2005','2006','2007','2008',
                        '2009','2010','2011','2012','2013','2014']
    data_drop = data[columns_needed].copy()

    # Transposing the DataFrame to swap rows and columns
    data_drop_t = data_drop.transpose()
    
    # Setting the first row as the new column headers    
    data_drop_t.columns = data_drop_t.iloc[0]
    
    # Removing the first row (original column headers) from the DataFrame
    data_drop_t = data_drop_t.iloc[1:]
    
    # Converting the index (years) to numeric type
    data_drop_t.index = pd.to_numeric(data_drop_t.index)
    
    # Adding a new column 'Years' and assigning the numeric index to it
    data_drop_t['Years'] = data_drop_t.index

    return data_drop, data_drop_t
      
    
def slice_data(data1):
    '''Slice the input DataFrame and retain only the 'Country Name' and '2014' 
    columns.
    Parameters:
    - data1 (DataFrame): The input DataFrame to be sliced.
    Returns:
    - data1 (DataFrame): The sliced DataFrame with 'Country Name' and '2014' 
    columns.'''
    
    # Slicing the DataFrame to include only 'Country Name' and '2014' columns
    data1=data1[['Country Name', '2014']]
    
    # Return the final merged DataFrame
    return data1

def merge_data(a1, a2, a3, a4):
    '''Merge multiple DataFrames based on the 'Country Name' column using outer joins.
    Parameters:
    - a1, a2, a3, a4 (DataFrames): Input DataFrames to be merged.
    Returns:
    - merged_a3 (DataFrame): The merged DataFrame containing data from all input 
    DataFrames.'''
    
    # Perform the first merge between a1 and a2
    merged_a1 = pd.merge(a1, a2, on='Country Name', how='outer')
    
    # Perform the second merge between the result of the first merge and a3
    merged_a2 = pd.merge(merged_a1, a3, on='Country Name', how='outer')
    
    # Perform the third merge between the result of the second merge and a4
    merged_a3 = pd.merge(merged_a2, a4, on='Country Name', how='outer')
    
    # Reset the index to ensure a clean DataFrame structure
    merged_a3 = merged_a3.reset_index(drop=True)
    
    # Return the final merged DataFrame
    return merged_a3

def correlation_heatmap(df):
    '''Create a correlation heatmap for the numeric columns in a DataFrame.
    Parameters:
    - df (DataFrame): The input DataFrame containing numeric columns.
    This function extracts numeric columns from the DataFrame, computes the 
    correlation matrix,and generates a heatmap using seaborn. 
    The heatmap is saved as 'Heatmap.png' and displayed.
    Parameters:
    - df (DataFrame): The input DataFrame containing numeric columns.'''

    # Extract numeric columns from the DataFrame for correlation analysis
    numeric_df = df.select_dtypes(include='number')
    
    # Compute the correlation matrix for the numeric columns
    correlation_matrix = numeric_df.corr()
    
    # Set the size of the heatmap figure
    plt.figure(figsize=(6, 5))
    
    # Create a heatmap using seaborn with annotationsand specified min-max values
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', vmin=-1, vmax=1)
    
    # Set the title for the heatmap
    plt.title('Correlation Matrix')
    
    # Save the heatmap as an image file
    plt.savefig('Heatmap.png')
    
    # Display the heatmap
    plt.show()
    
def line_plot(line_plot_data, title, x_label, y_label, legend_labels):
    
    '''Create a line plot from the provided DataFrame.
    Parameters:
    - line_plot_data (DataFrame): The input DataFrame containing data for the 
    line plot.
    - title (str): The title of the line plot.
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    - legend_labels (list): A list of labels for the legend.'''
    
    # Create a figure for the line plot
    plt.figure()
    
    # Plot the data with specified settings
    line_plot_data.plot(x='Years', y=legend_labels, figsize=(9,5),
                            linestyle='dashed', marker='*', markersize = 10)  

    # Set the title, x-axis label, and y-axis label                  
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Add a legend with specified labels
    plt.legend(legend_labels)
    
    # Save the line plot as an image file
    plt.savefig('Lineplot.png')
    
    # Display the line plot
    plt.show()
    

def bar_plot(df, x_value, y_values, head_title, x_label, y_label,
             colors, figsize=(10, 6)):
    
    '''Create a bar plot from the provided DataFrame.
    Parameters:
    - df (DataFrame): The input DataFrame containing data for the bar plot.
    - x_value (str): The column to be used for the x-axis.
    - y_values (list): A list of columns to be plotted on the y-axis.
    - head_title (str): The title of the bar plot.
    - x_label (str): The label for the x-axis.
    - y_label (str): The label for the y-axis.
    - colors (list): A list of colors for each bar.
    - figsize (tuple, optional): The size of the figure (default is (10, 6)).'''
    
    # Set the seaborn style to whitegrid
    sns.set_style('whitegrid') 
    
    # Filter the DataFrame for specific years (2004, 2006, 2008, 2010, 2012, 2014)
    df_filtered = df[df['Years'].isin([2004, 2006, 2008, 2010, 2012, 2014])]

    # Create a bar plot using the filtered DataFrame
    df_filtered.plot(x=x_value, y=y_values, kind='bar', title=head_title,
                     color=colors, width=0.45, figsize=figsize, xlabel=x_label,
                     ylabel=y_label)
    
    # Add a legend to the plot
    plt.legend(loc='best', bbox_to_anchor=(1, 0.4))
    
    # Save the bar plot as an image file
    plt.savefig('Borplot.png')
    
    # Display the bar plot
    plt.show()
    
def box_plot(data, countries, title, x_label="Country",
                    y_label="CO2 Emission (kt)", figsize=(7, 5)):
    
    '''Create a box plot to visualize CO2 emissions for multiple countries.

    Parameters:
    - data (dict): A dictionary containing CO2 emission data for each country.
    - countries (list): A list of countries to be included in the box plot.
    - title (str): The title of the box plot.
    - x_label (str, optional): The label for the x-axis (default is "Country").
    - y_label (str, optional): The label for the y-axis (default is "CO2 Emission 
                                                         (kt)").
    - figsize (tuple, optional): The size of the figure (default is (7, 5)).'''
   
    # Set the seaborn style to whitegrid
    sns.set(style="whitegrid")
    
    # Create a figure for the box plot
    plt.figure(figsize=figsize)

    # Convert the data dictionary to a Pandas DataFrame
    df = pd.DataFrame({country: data[country] for country in countries})

    # Create box plot
    sns.boxplot(data=df)

    # Set the x-axis label, y-axis label, and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Save the box plot as an image file
    plt.savefig('Boxplot.png')
    
    # Display the box plot
    plt.show()
    
def pie_plot(df, year, autopct='%1.0f%%', fontsize=11):
    
    '''Create a pie plot to visualize population distribution for specific 
    countries.
    Parameters:
    - df (DataFrame): The input DataFrame containing population data.
    - year (int): The specific year for which the population distribution is 
    visualized.
    - autopct (str, optional): The format for displaying autopct on pie slices 
    (default is '%1.0f%%').
    - fontsize (int, optional): Font size for the title (default is 11).'''
    
    # Define parameters for pie plot
    explode = (0.0, 0.0, 0.05, 0.0, 0.0)
    labels = ["Germany","France","Japan","Netherlands","Thailand"]
    
    # Create a figure for the pie plot
    plt.figure(figsize=(4, 5))
    
    # Create the pie plot using matplotlib
    plt.pie(df[str(year)],
            autopct=autopct, labels=labels, explode=explode,
            startangle=180, wedgeprops={"edgecolor": "black",
                                        "linewidth": 2, "antialiased": True})
    
    # Set the title for the pie plot
    plt.title(f'Population in {year}', fontsize=fontsize)
    
    # Save the pie plot as an image file
    plt.savefig('pieplot.png')
    
    # Display the pie plot
    plt.show()
    
def skewness_kurtosis(df):
    '''Calculate skewness and kurtosis of a given DataFrame.
    Parameters:
    - df (DataFrame): Input DataFrame for which skewness and kurtosis are to be 
    calculated.
    Returns:
    - tuple: A tuple containing two pandas Series representing skewness and 
    kurtosis,'''
    
    # Calculate the skewness and kurtosis of the given DataFrame
    skewness=df.skew()
    kurt=df.kurtosis()
    
    # Return the calculated skewness and kurtosis
    return skewness,kurt
    
# File path to the data file
file_path = "API_19_DS2_en_csv_v2_6224512.csv"

# List of countries to analyze
countries_list = ["Germany","France","Japan","Netherlands","Thailand"]

# Process data for Population
dt1, dt1_t = read_and_process_data(file_path, "Population, total",
                                       countries_list)

# Process data for Nitrous oxide emissions
dt2, dt2_t = read_and_process_data(
    file_path,
    "Nitrous oxide emissions (thousand metric tons of CO2 equivalent)",
    countries_list)

# Process data for CO2 emissions from gaseous fuel consumption
dt3, dt3_t = read_and_process_data(
    file_path, "CO2 emissions from gaseous fuel consumption (kt)",
    countries_list)

# Process data for Electric power consumption
dt4, dt4_t = read_and_process_data(
    file_path, "Electric power consumption (kWh per capita)", countries_list)

# Slice and rename columns for Population total in 2014
slice_data1=slice_data(dt1).rename(columns={'2014': 'Population total'})

# Slice and rename columns for Nitrous oxide emissions in 2014
slice_data2=slice_data(dt2).rename(columns={'2014': 'Nitrous oxide emissions'})

# Slice and rename columns for CO2 emissions in 2014
slice_data3=slice_data(dt3).rename(columns={'2014': 'CO2 emissions'})

# Slice and rename columns for Electric power consumption in 2014
slice_data4=slice_data(dt4).rename(columns={'2014': 'Electric power consumption'})

# Merge sliced and renamed DataFrames into a single DataFrame
merged_data = merge_data(slice_data1, slice_data2, slice_data3, slice_data4)

# Display summary statistics for Population data (data1)
print(dt1.describe())

# Create a correlation heatmap for the merged data
correlation_heatmap(merged_data)

# Create a line plot for Nitrous Oxide Emission rates over the years
line_plot(dt3_t, 'Nitrous Oxide Emission', 'Years', 'Emission rate', 
          ["Germany","France","Japan","Netherlands","Thailand"])

# Create a bar plot for Electric Power Consumption over selected years
bar_plot(dt4_t, 'Years',
         ["Germany","France","Japan","Netherlands","Thailand"],
         'Electric Power Consumption (kWh per capita) - Selected years',
         'Years', 'Electric Power Consumption rate(kWh per capita)',
         ['lightcoral', 'skyblue', 'blue', 'green', 'black'])

# Create a box plot for CO2 Emissions from Gaseous Fuel Consumption
box_plot(dt2_t, ["Germany","France","Japan","Netherlands","Thailand"],
                'CO2 Emissions from Gaseous Fuel Consumption')

# Create a pie plot for Population distribution in 2013
pie_plot(dt1, 2013)

# Calculate skewness and kurtosis for data in the 'Japan' column of DataFrame
skewness,kurtosis=skewness_kurtosis(dt2_t['Japan'])

# Print the calculated skewness and kurtosis
print(skewness)
print(kurtosis)


