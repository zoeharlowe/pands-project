# analysis.py
# This program contains all the code for the analysis of the Iris dataset.
# Author: Zoe McNamara Harlowe

# 1. Importing libraries:

# Dataframes
import pandas as pd

# Numpy
import numpy as np

# ScikitLearn: Machine Learning repository that contains sample datasets
import sklearn as skl 
from sklearn import datasets

# Plots
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Seaborn: Data visualization library based on matplotlib
import seaborn as sns

# Scipy: Scientific library for Python
import scipy

# 2. Loading the dataset:

# Set column names
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Read in the dataset 
df = pd.read_csv("iris.csv", names = column_names)

# Display first 5 rows of the dataset
df.head(5)

# Display the last 5 rows of the dataset
df.tail(5)

# 3. Describe the dataset:

# Display feature names
column_names = df.columns.values
print(column_names)

# Find unique values in the species column.
unique_values = np.unique(df["species"])
print(unique_values)

# Display summary statistics of the dataset
df.describe()

# 4. Create summary file for the dataset:

# Create file
FILENAME = "summary.txt"

with open(FILENAME, 'w') as f:

    # Title
    f.write("Iris Dataset Summary\n")
    f.write("====================================\n\n")

    # Overall summary
    f.write("OVERALL SUMMARY\n")
    f.write(f"Shape of dataset: \t {df.shape} \n") # shape
    f.write(f"Number of species: \t {len(unique_values)} \n") # number of species

    # Number of features - I created a lambda function to count the number of features in each row
    float_count = df.apply(lambda row: sum(isinstance(x, float) for x in row), axis=1).iloc[0]
    f.write(f"Number of features:  {float_count} \n") # number of features
    
    f.write(f"Species names:\t\t {unique_values} \n") # species names
    f.write(f"Feature names:\t\t {column_names} \n") # variable names

# Features summary
# Open file in append mode to avoid overwriting the previous content
SUMMARY_FILE = "summary.txt"
with open(SUMMARY_FILE, 'a') as f:
    # Sepal length
    f.write("\nSEPAL LENGTH SUMMARY\n")
    f.write(df["sepal_length"].describe().to_string() + "\n")

    # Sepal width
    f.write("\nSEPAL WIDTH SUMMARY\n")
    f.write(df["sepal_width"].describe().to_string() + "\n")

    # Petal length
    f.write("\nPETAL LENGTH SUMMARY\n")
    f.write(df["petal_length"].describe().to_string() + "\n")

    # Petal width
    f.write("\nPETAL WIDTH SUMMARY\n")
    f.write(df["petal_width"].describe().to_string() + "\n")

    # Species
    f.write("\nSPECIES SUMMARY\n")
    f.write(df["species"].describe().to_string() + "\n")

# 5. Explore the dataset:

# Creating arrays for each feature and species
# Sepal width
sepal_width = df["sepal_width"]
sepal_width = sepal_width.to_numpy()

# Sepal length
sepal_length = df["sepal_length"]
sepal_length = sepal_length.to_numpy()

# Petal width
petal_width = df["petal_width"]
petal_width = petal_width.to_numpy()

# Petal length
petal_length = df["petal_length"]
petal_length = petal_length.to_numpy()

# Species
species = df["species"]
species = species.to_numpy()

# I then created multiple series of numpy arrays for each species and each feature.
# I used pandas dataframe to filter the data by species
# Setosa sepal width
setosa_sepal_width = df[df["species"] == "Iris-setosa"]["sepal_width"].to_numpy()

# Versicolor sepal width
versicolor_sepal_width = df[df["species"] == "Iris-versicolor"]["sepal_width"].to_numpy()

# Virginica sepal width
virginica_sepal_width = df[df["species"] == "Iris-virginica"]["sepal_width"].to_numpy()

# Setosa sepal length
setosa_sepal_length = df[df["species"] == "Iris-setosa"]["sepal_length"].to_numpy()

# Versicolor sepal length
versicolor_sepal_length = df[df["species"] == "Iris-versicolor"]["sepal_length"].to_numpy()

# Virginica sepal length
virginica_sepal_length = df[df["species"] == "Iris-virginica"]["sepal_length"].to_numpy()

# Setosa petal width
setosa_petal_width = df[df["species"] == "Iris-setosa"]["petal_width"].to_numpy()

# Versicolor petal width
versicolor_petal_width = df[df["species"] == "Iris-versicolor"]["petal_width"].to_numpy()

# Virginica petal width
virginica_petal_width = df[df["species"] == "Iris-virginica"]["petal_width"].to_numpy()

# Setosa petal length
setosa_petal_length = df[df["species"] == "Iris-setosa"]["petal_length"].to_numpy()

# Versicolor petal length
versicolor_petal_length = df[df["species"] == "Iris-versicolor"]["petal_length"].to_numpy()

# Virginica petal length
virginica_petal_length = df[df["species"] == "Iris-virginica"]["petal_length"].to_numpy()

# 5a. Histograms:
# Histogram of sepal width with set colour customisation and bin width
plt.hist(sepal_width, bins=10, color='blue', edgecolor='black')

# Add axis labels
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# Add title
plt.title('Sepal Width Distribution')

# Save the histogram to a file
plt.savefig("histogram_sepal_width.png")

# Show
plt.show()

# Histogram of sepal length with set colour customisation and bin width
plt.hist(sepal_length, bins=10, color='red', edgecolor='black')

# Add axis labels
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')

# Add title
plt.title('Sepal Length Distribution')

# Save the histogram to a file
plt.savefig("histogram_sepal_length.png")

# Show
plt.show()

# Histogram of petal width with set colour customisation and bin width
plt.hist(petal_width, bins=10, color='green', edgecolor='black')

# Add axis labels
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')

# Add title
plt.title('Petal Width Distribution')

# Save the histogram to a file
plt.savefig("histogram_petal_width.png")

# Show
plt.show()

# Histogram of petal length with set colour customisation and bin width
plt.hist(petal_length, bins=10, color='yellow', edgecolor='black')

# Add axis labels
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

# Add title
plt.title('Petal Length Distribution')

# Save the histogram to a file
plt.savefig("histogram_petal_length.png")

# Show
plt.show()

# Histogram of species with set colour customisation and bin width
plt.hist(species, bins=10, color='gray', edgecolor='black')

# Add axis labels
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

# Add title
plt.title('Species Distribution')

# Save the histogram to a file
plt.savefig("histogram_species.png")

# Show
plt.show()

# 5b. Boxplots:

# Boxplot of sepal width with set colour customisation and bin width
# List of features to represent in boxplot
data_to_plot = [setosa_sepal_width, versicolor_sepal_width, virginica_sepal_width]

# Create boxplot
bp = plt.boxplot(data_to_plot, patch_artist=True, labels=["Setosa", "Versicolor", "Virginica"])

# Title
plt.title("Boxplot of Sepal Width by Species")

# Axis labels
plt.ylabel("Sepal Width (cm)", fontweight='bold')
plt.xlabel("Species Names", fontweight='bold')

# Box colours
# Set colours using a list
colors = ["magenta", "gold", "purple"]

# ChatGPT to apply colours: https://chatgpt.com/share/68090395-151c-8000-b018-be5c812b4ee9
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='darkblue',
              alpha = 0.5)

# Changing colour and linewidth of caps
for cap in bp['caps']:
    cap.set(color ='darkgreen',
            linewidth = 2)
    
# Changing colour and linewidth of median
for median in bp['medians']:
    median.set(color ='lawngreen',
               linewidth = 1.5)
    
# Save the boxplot to a file
plt.savefig("boxplot_sepal_width.png")

# Boxplot of sepal length with set colour customisation and bin width
# List of features to represent in boxplot
data_to_plot = [setosa_sepal_length, versicolor_sepal_length, virginica_sepal_length]

# Create boxplot
bp = plt.boxplot(data_to_plot, patch_artist=True, labels=["Setosa", "Versicolor", "Virginica"])

# Title
plt.title("Boxplot of Sepal Length by Species")

# Axis labels
plt.ylabel("Sepal Length (cm)", fontweight='bold')
plt.xlabel("Species Names", fontweight='bold')

# Box colours
# Set colours using a list
colors = ["magenta", "gold", "purple"]

# ChatGPT to apply colours: https://chatgpt.com/share/68090395-151c-8000-b018-be5c812b4ee9
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='darkblue',
              alpha = 0.5)

# Changing colour and linewidth of caps
for cap in bp['caps']:
    cap.set(color ='darkgreen',
            linewidth = 2)
    
# Changing colour and linewidth of median
for median in bp['medians']:
    median.set(color ='lawngreen',
               linewidth = 1.5)
    
# Save the boxplot to a file
plt.savefig("boxplot_sepal_length.png")

# Show.
plt.show()

# Boxplot of petal width with set colour customisation and bin width
# List of features to represent in boxplot
data_to_plot = [setosa_petal_width, versicolor_petal_width, virginica_petal_width]

# Create boxplot
bp = plt.boxplot(data_to_plot, patch_artist=True, labels=["Setosa", "Versicolor", "Virginica"])

# Title
plt.title("Boxplot of Petal Width by Species")

# Axis labels
plt.ylabel("Sepal Length (cm)", fontweight='bold')
plt.xlabel("Species Names", fontweight='bold')

# Box colours
# Set colours using a list
colors = ["magenta", "gold", "purple"]

# ChatGPT to apply colours: https://chatgpt.com/share/68090395-151c-8000-b018-be5c812b4ee9
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='darkblue',
              alpha = 0.5)

# Changing colour and linewidth of caps
for cap in bp['caps']:
    cap.set(color ='darkgreen',
            linewidth = 2)
    
# Changing colour and linewidth of median
for median in bp['medians']:
    median.set(color ='lawngreen',
               linewidth = 1.5)
    
# Save the boxplot to a file
plt.savefig("boxplot_petal_width.png")

# Show.
plt.show()

# Boxplot of petal length with set colour customisation and bin width
# List of features to represent in boxplot
data_to_plot = [setosa_petal_length, versicolor_petal_length, virginica_petal_length]

# Create boxplot
bp = plt.boxplot(data_to_plot, patch_artist=True, labels=["Setosa", "Versicolor", "Virginica"])

# Title
plt.title("Boxplot of Petal Length by Species")

# Axis labels
plt.ylabel("Petal Length (cm)", fontweight='bold')
plt.xlabel("Species Names", fontweight='bold')

# Box colours
# Set colours using a list
colors = ["magenta", "gold", "purple"]

# ChatGPT to apply colours: https://chatgpt.com/share/68090395-151c-8000-b018-be5c812b4ee9
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='darkblue',
              alpha = 0.5)

# Changing colour and linewidth of caps
for cap in bp['caps']:
    cap.set(color ='darkgreen',
            linewidth = 2)
    
# Changing colour and linewidth of median
for median in bp['medians']:
    median.set(color ='lawngreen',
               linewidth = 1.5)
    
# Save the boxplot to a file
plt.savefig("boxplot_petal_length.png")

# Show.
plt.show()

# 5c. Scatterplots:

# Scatterplot of sepal length vs sepal width
# Scatterplot with a different colour for each class type.
# I used the fig, ax method so I can recall the same scatterplot in later tasks.
fig1, ax = plt.subplots()
ax.scatter(setosa_sepal_length, setosa_sepal_width, marker = '.', c = 'magenta', label = "Setosa")
ax.scatter(versicolor_sepal_length, versicolor_sepal_width, marker = '.', c = 'gold', label = "Versicolor")
ax.scatter(virginica_sepal_length, virginica_sepal_width, marker = '.', c = 'purple', label = "Virginica")

# Title.
ax.set_title('Comparison of sepal length and sepal width for each flower in the Iris dataset')

# Axes labels.
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')

ax.legend()

plt.savefig("scatterplot_sepal_length_sepal_width.png")

# Show.
plt.show()

# Scatterplot of petal length vs petal length
# Scatterplot with a different colour for each class type.
# I used the fig, ax method so I can recall the same scatterplot in later tasks.
fig2, ax = plt.subplots()
ax.scatter(setosa_sepal_length, setosa_petal_length, marker = '.', c = 'magenta', label = "Setosa")
ax.scatter(versicolor_sepal_length, versicolor_petal_length, marker = '.', c = 'gold', label = "Versicolor")
ax.scatter(virginica_sepal_length, virginica_petal_length, marker = '.', c = 'purple', label = "Virginica")

# Title.
ax.set_title('Comparison of sepal length and petal length for each flower in the Iris dataset')

# Axes labels.
ax.set_xlabel('Sepal length')
ax.set_ylabel('Petal length')

ax.legend()

plt.savefig("scatterplot_sepal_length_petal_length.png")

# Show.
plt.show()

# Scatterplot of sepal length vs petal width
# Scatterplot with a different colour for each class type.
# I used the fig, ax method so I can recall the same scatterplot in later tasks.
fig3, ax = plt.subplots()
ax.scatter(setosa_sepal_length, setosa_petal_width, marker = '.', c = 'magenta', label = "Setosa")
ax.scatter(versicolor_sepal_length, versicolor_petal_width, marker = '.', c = 'gold', label = "Versicolor")
ax.scatter(virginica_sepal_length, virginica_petal_width, marker = '.', c = 'purple', label = "Virginica")

# Title.
ax.set_title('Comparison of sepal length and petal width for each flower in the Iris dataset')

# Axes labels.
ax.set_xlabel('Sepal length')
ax.set_ylabel('Petal width')

ax.legend()

plt.savefig("scatterplot_sepal_length_petal_width.png")

# Show.
plt.show()

# Scatterplot of petal length vs sepal width
# Scatterplot with a different colour for each class type.
# I used the fig, ax method so I can recall the same scatterplot in later tasks.
fig4, ax = plt.subplots()
ax.scatter(setosa_petal_length, setosa_sepal_width, marker = '.', c = 'magenta', label = "Setosa")
ax.scatter(versicolor_petal_length, versicolor_sepal_width, marker = '.', c = 'gold', label = "Versicolor")
ax.scatter(virginica_petal_length, virginica_sepal_width, marker = '.', c = 'purple', label = "Virginica")

# Title.
ax.set_title('Comparison of sepal length and petal length for each flower in the Iris dataset')

# Axes labels.
ax.set_xlabel('Petal length')
ax.set_ylabel('Sepal width')

ax.legend()

plt.savefig("scatterplot_petal_length_sepal_width.png")

# Show.
plt.show()

# Scatterplot of petal length vs petal width
# Scatterplot with a different colour for each class type.
# I used the fig, ax method so I can recall the same scatterplot in later tasks.
fig5, ax = plt.subplots()
ax.scatter(setosa_petal_length, setosa_petal_width, marker = '.', c = 'magenta', label = "Setosa")
ax.scatter(versicolor_petal_length, versicolor_petal_width, marker = '.', c = 'gold', label = "Versicolor")
ax.scatter(virginica_petal_length, virginica_petal_width, marker = '.', c = 'purple', label = "Virginica")

# Title.
ax.set_title('Comparison of petal length and petal width for each flower in the Iris dataset')

# Axes labels.
ax.set_xlabel('Petal length')
ax.set_ylabel('Petal width')

ax.legend()

plt.savefig("scatterplot_petal_length_petal_width.png")

# Show.
plt.show()

# Scatterplot of sepal width vs petal width
# Scatterplot with a different colour for each class type.
# I used the fig, ax method so I can recall the same scatterplot in later tasks.
fig6, ax = plt.subplots()
ax.scatter(setosa_sepal_width, setosa_petal_width, marker = '.', c = 'magenta', label = "Setosa")
ax.scatter(versicolor_sepal_width, versicolor_petal_width, marker = '.', c = 'gold', label = "Versicolor")
ax.scatter(virginica_sepal_width, virginica_petal_width, marker = '.', c = 'purple', label = "Virginica")

# Title.
ax.set_title('Comparison of sepal width and petal width for each flower in the Iris dataset')

# Axes labels.
ax.set_xlabel('Sepal width')
ax.set_ylabel('Petal width')

ax.legend()

plt.savefig("scatterplot_sepal_width_petal_width.png")

# Show.
plt.show()

# 5d. Correlation matrix:

# Define the features which I will be finding the correlation coefficient of.
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Identify the correlation coefficient between all four features.
# This will return a correlation matrix.
correlation_matrix = df[features].corr()

print(correlation_matrix)

# 5e. Heatmap:

# Identify the features being compared in the heatmap.
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# Define custom color sequence (dark red → red → orange → yellow → white → yellow → orange → red → dark red)
colors = [
    (0.4, 0, 0),   # dark red
    (1, 0, 0),     # red
    (1, 0.5, 0),   # orange
    (1, 1, 0),     # yellow
    (1, 1, 1),     # white
    (1, 1, 0),     # yellow
    (1, 0.5, 0),   # orange
    (1, 0, 0),     # red
    (0.4, 0, 0)    # dark red
]

# Give the colormap a name
cmap_name = 'cyclic_autumn'

# Create a colormap from the list of colors
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256) # N=256 for smooth gradient

# Create heatmap.
plt.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1) # Set colour scale from -1 to 1

# Add title
plt.title("Correlation coefficients of the 4 features in Iris dataset")

# Set tick labels
plt.xticks(range(len(features)),
           # Rotate the x-axis labels by 40 degrees for better visibility
           features, rotation=40) 
plt.yticks(range(len(features)), 
           features) 

# Annotate each cell using a for loop.
for i in range(len(features)):
    for j in range(len(features)):
        # Round each correlation coefficient to two decimal places.
        value = round(correlation_matrix.iloc[i, j], 2)
        # Format each annotation to be in the centre of the cell, coloured black and in bold.
        plt.text(j, i, str(value), ha='center', va='center', color='black', fontweight = 'bold')

# Set axes labels and format them in bold
plt.ylabel("Features", fontweight = 'bold')
plt.xlabel("Features", fontweight = 'bold')

# Add colourbar 
cbar = plt.colorbar(ticks=[-1, 0, 1]) 
# Set colourbar labels
cbar.ax.set_yticklabels(['-1 (Perfectly Negative Linear Correlation)', '0 (No Correlation)', '1 (Perfectly Positive Linear Correlation)'])

# Save the heatmap to a file
plt.savefig("heatmap_correlation.png")

# Show.
plt.show()

# f. Pairplot:

# Load the pairplot using Seaborn
# Plot colours to the species of Iris flower using the 'hue' argument
# Make diagonal plots KDEs instead of histograms
g = sns.pairplot(df, hue="species", diag_kind='kde')

# Add title
# Add some height to the title using the 'y' argument as it was covering some of the pairplot
g.figure.suptitle("Pairplot of Iris Dataset", y=1)

# Save the pairplot to a file
g.savefig("pairplot.png")

# Show.
plt.show()