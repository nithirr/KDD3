from save_load import *
from confu import *
import seaborn as sns
y_test=load("y_test")
y_pred=load("y_pred")
res=confu_matrix(y_test,y_pred)


import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ["accuracy", "precision", "sensitivity", "specificity", "f_measure", "mcc", "npv", "fpr", "fnr"]
categories = [category.capitalize() for category in categories]
values = [i*100 for i in res]

fig, ax = plt.subplots()

# Create the lollipop chart
(markerline, stemlines, baseline) = ax.stem(range(len(categories)), values, basefmt=" ")


plt.setp(markerline, 'marker', 'o', 'color', 'red')
plt.setp(stemlines, 'color', 'green')

# Customize the plot
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_xlabel('Matrices')
ax.set_ylabel('Values')

# Show the plot
plt.tight_layout()
plt.savefig("Results/met",dpi=400)
plt.show()

def res_plot2():
    mat = confusion_matrix(load("y_test"), load("y_pred"))

    #df = pd.read_csv("data.csv")
    #df = df.iloc[:, -1]
    #print(df.unique())

    # Plot confusion matrix
    plt.figure(figsize=(12, 6))
    sns.heatmap(mat, annot=True, cmap="OrRd", fmt="d", xticklabels=['Normal','Tumor'],yticklabels=['Normal','Tumor'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig("Results/confu.png", dpi=400)
    plt.show()

res_plot2()

import pandas as pd
metrics_df = pd.DataFrame({'Metric': categories, 'Value': values})
metrics_df.to_csv("Results/res.csv")
# Display the DataFrame
print(metrics_df)

def plot2():

    data = {
        "Methods": ["accuracy", "precision", "sensitivity", "specificity", "f_measure", "mcc", "npv", "fpr", "fnr"],
        "values": [i * 100 for i in res]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.barplot(y="Methods", x="values", data=df, palette="cividis")

    plt.xlabel('values')
    plt.ylabel('Metrices')

    # Display the plot
    plt.savefig("Results/met1",dpi=400)
    plt.show()
plot2()
