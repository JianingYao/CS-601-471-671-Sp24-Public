import numpy as np
import matplotlib.pyplot as plt

# creating the dataset
data = {
        'distilbert-base-uncased-dev': 71.19, 'distilbert-base-uncased-test': 69.24,
        'roberta-base-dev': 77.89, 'roberta-base-test': 76.03
        }
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, color='maroon',
        width=0.4)

plt.ylabel("Accuracy (%)")
plt.xlabel("Models_on_Dataset")
plt.title("Model Performance")
plt.savefig('bar_plot.png')
