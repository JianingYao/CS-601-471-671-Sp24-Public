import numpy as np
import matplotlib.pyplot as plt

# creating the dataset
data = {
        'full': 98.66, 'head': 69.24, 'prefix': 189.56}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, color='maroon',
        width=0.4)

plt.ylabel("CUDA Mem (Mb)")
plt.xlabel("Models")
plt.title("CUDA Mem Usage")
plt.savefig('bar_plot.png')
