# pands-project
# Author: Zoe McNamara Harlowe

# Table of Contents
- Introduction
- Setup
- Tasks
- Summary of the Iris dataset
- About the Author
- References

# 1. Introduction
### This repository contains my project submission for the Programming and Scripting module as part of the ATU Higher Diploma in Computing in Data Analytics. This project explores the Iris dataset, which is often used as a beginner's dataset in data analytics.

# 2. Setup 

##### Follow the steps below to set up this project locally using **Visual Studio Code** and **Git**. You need to have Visual Studio Code, as well as the VSCode Python and Jupyter Extensions, downloaded before beginning these steps. This guide is for Windows users.

### 1. Clone the Repository
##### First, open a terminal and run:

```bash
git clone https://github.com/zoeharlowe/pands-project.git
cd pands-project
```
### 2. Open the project in VSCode

``` bash
code .
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the Files

##### Launch Jupyter Notebook or JupyterLab from the terminal:

``` bash
jupyter notebook
```

##### Then open the notebook file or the code file:
- In the Explorer panel (left sidebar), click on `iris.ipynb` to open the Jupyter Notebook.
- If you want to just look at the code, click on `analysis.py`.

# 3. Tasks
#### In this project, I write a program called analysis.py that:  
- will **load** the Iris dataset using Pandas. 
- will **describe** the Iris dataset and identify the feature names and species names.
- will output the **summary** of each variable into a single text file.
- will create a **histogram** for each feature. 
- will create a **boxplot** for each feature. 
- will create a **scatterplot** for each pair of features.
- will calculate the **correlation coefficient** between the four features.
- will create a **heatmap** to display the correlation coefficients of each feature with one another. 
- will create a **pairplot** to display the relationship between all features with one another.

#### You can simply just view my code by clicking into `analysis.py`, or else view the Jupyter Notebook which contains the code along with my understandings and assumptions in `iris.ipynb`.

# 4. Summary of the Iris dataset
##### The **Iris dataset** is a classic and widely used dataset in the field of machine learning and statistics. It contains **150 observations** of iris flowers, with **four numerical features** and a **categorical class label** representing the species.

![alt text](R.png)

[1] Photo from Analytics Vidhya: https://th.bing.com/th/id/R.9df2b7bc66644dae06aceef25a9712f9?rik=mrzdphMfvHvc0A&pid=ImgRaw&r=0

### Features
##### Each observation includes the following features:

- **Sepal Length** (in cm)
- **Sepal Width** (in cm)
- **Petal Length** (in cm)
- **Petal Width** (in cm)

##### These features are continuous variables representing physical characteristics of the iris flowers.

### Species
##### The dataset includes samples from **three different species** of Iris:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

##### Each species has **50 samples**.

### Purpose
##### The Iris dataset is often used for:

- **Classification** tasks (e.g., predicting species based on measurements)
- **Clustering** and **unsupervised learning**
- **Visualization** techniques (e.g., scatterplots, pair plots)
  
### Source
##### Originally introduced by **Ronald A. Fisher** in 1936, the dataset is available in many libraries including:

- [Scikit-learn](https://scikit-learn.org/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

###### [2] *https://www.geeksforgeeks.org/iris-dataset/*
###### [3] *https://chatgpt.com/share/681f696b-ed80-8000-a52d-4f8c0f8760e3*

# 5. About the Author
##### My name is Zoe McNamara Harlowe and I am from Limerick, Ireland. I have just begun my journey in data analytics this year. I am looking to improve my ability to code as well as explore the fascinating area of data analysis. I have a degree in Education with Irish and Spanish and I am interested in finding out about the ways that I can apply data analytics to this field. Since starting this Postgrad in January, I have thoroughly enjoyed gaining valuable knowledge about this area, and I am looking forward to learning more about Python, machine learning and statistical analysis in future modules.

# 6. References
- [1] Photo from Analytics Vidhya: https://th.bing.com/th/id/R.9df2b7bc66644dae06aceef25a9712f9?rik=mrzdphMfvHvc0A&pid=ImgRaw&r=0

- [2] GeeksForGeeks article about Iris dataset: https://www.geeksforgeeks.org/iris-dataset/

- [3] ChatGPT: Help with colormaps and README: https://chatgpt.com/share/681f696b-ed80-8000-a52d-4f8c0f8760e3

- [4] Raw Iris dataset: https://archive.ics.uci.edu/dataset/53/iris

- [5] Pandas documentation on reading in CSV files: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

- [6] GeeksForGeeks article about df.head() and df.tail(): https://www.geeksforgeeks.org/difference-between-pandas-head-tail-and-sample/

- [7] Pandas documentation on df.columns(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html#pandas.DataFrame.columns

- [8] Numpy documentation on np.unique(): https://numpy.org/doc/stable/reference/generated/numpy.unique.html

- [9] Pandas documentation on df.describe(): https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html

- [10] GeeksForGeeks article on creating new text files: https://www.geeksforgeeks.org/create-a-new-text-file-in-python/

- [11] ChatGPT: Create lambda function to count number of features: https://chatgpt.com/share/68023853-726c-8000-901f-72d720dfc9bf

- [12] ChatGPT: Question about f.write() and learning about to_string(): https://chatgpt.com/share/680b7005-0a00-8000-b664-43050f0d49e8

- [13] Pandas documentation on to_numpy(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html

- [14] ChatGPT: Create arrays for each species and feature combination: https://chatgpt.com/share/681a689e-af98-8000-b1fe-cc017fc24afa

- [15] Matplotlib documentation on creating histograms: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html 

- [16] Statology article about bimodal distribution: https://www.statology.org/bimodal-distribution/

- [17] Matplotlib documentation on creating boxplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html

- [18] GeeksForGeeks article about displaying multiple datasets in one boxplot: https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/

- [19] ChatGPT: Applying colours to boxplot using for loops: https://chatgpt.com/share/68090395-151c-8000-b018-be5c812b4ee9

- [20] Article about interpreting boxplots: https://www.vrogue.co/post/understanding-boxplots-how-to-read-and-interpret-a-boxplot-built-in

- [21] Matplotlib documentation on creating scatterplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

- [22] Pandas documentation on finding correlation coefficients with df.corr(): https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

- [23] Statology article about correlation in Python: https://www.statology.org/correlation-in-python/

- [24] Matplotlib documentation on annotated heatmaps: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

- [25] ChatGPT: Create a custom colormap for a heatmap: https://chatgpt.com/share/67ffaf4d-5d08-8000-acf8-bd59f1760cbe

- [26] Seaborn documentation on creating pairplots: https://seaborn.pydata.org/generated/seaborn.pairplot.html

- [27] GeeksForGeeks article about Seaborn pairplots: https://www.geeksforgeeks.org/python-seaborn-pairplot-method/

- [28] YouTube video about exploring options with pairplots/customisation: https://www.youtube.com/watch?v=cpZExlOKFH4

- [29] GeeksForGeeks article about interpreting pairplots: https://www.geeksforgeeks.org/data-visualization-with-pairplot-seaborn-and-pandas/
