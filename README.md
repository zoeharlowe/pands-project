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
### This repository contains my project submission for the Programming and Scripting module as part of the ATU Higher Diploma in Computing in Data Analytics. This project explores the Iris dataset, which was introduced by the statistician Ronald Fisher in 1936 and is often used as a beginner's dataset in data analytics.

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

# 4. Summary of the Iris dataset
##### The **Iris dataset** is a classic and widely used dataset in the field of machine learning and statistics. It contains **150 observations** of iris flowers, with **four numerical features** and a **categorical class label** representing the species.

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
- Demonstrating **data preprocessing** and **model evaluation**

### Source
##### Originally introduced by **Ronald A. Fisher** in 1936, the dataset is available in many libraries including:

- [Scikit-learn](https://scikit-learn.org/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- R datasets package


###### [1] *https://www.geeksforgeeks.org/iris-dataset/*
###### [2] Help with colormaps and README *https://chatgpt.com/share/681f696b-ed80-8000-a52d-4f8c0f8760e3*

# 5. About the Author
##### My name is Zoe McNamara Harlowe and I am from Limerick, Ireland. I have just begun my journey in data analytics this year. I am looking to improve my ability to code as well as explore the fascinating area of data analysis. I have a degree in Education with Irish and Spanish and am interested in seeing the ways that I can apply data analytics to this field. Since starting in January, I have thoroughly enjoyed gaining valuable knowledge about this area, and I am looking forward to further learning more about Python, machine learning and statistical analysis.

# 6. References
[1] GeeksForGeeks: Iris dataset https://www.geeksforgeeks.org/iris-dataset/
[2] ChatGPT: Help with colormaps and README https://chatgpt.com/share/681f696b-ed80-8000-a52d-4f8c0f8760e3
[3] 