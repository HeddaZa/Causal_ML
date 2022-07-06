'''Definitions used for plots'''

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

def pie_plots(data,treatment_name, target_name):
    treatments = data[treatment_name].unique()
    explode = (0, 0.1) 
    fig, axes = plt.subplots(1, 4,figsize=(14,7))
    fig.set_facecolor('lightgrey')

    axes[0].pie(
        data.groupby(target_name)[treatment_name].count(),
        textprops={'fontsize': 14}, 
            colors = ['lightsteelblue','darkseagreen'], 
            shadow = True,
            labels = ['no visit', 'visit'], 
            explode = explode,
            autopct='%1.0f%%'
        )
    axes[0].set_title('All treatments')
    for i, treatment in enumerate(treatments):
        axes[i+1].pie(
            data[data[treatment_name] == treatment].groupby(target_name)['recency'].count(),
            textprops={'fontsize': 14}, 
            colors = ['lightsteelblue','darkseagreen'], 
            shadow = True,
            labels = ['no visit', 'visit'], 
            explode = explode,
            autopct='%1.0f%%'
            )
        axes[i+1].set_title(treatment)