'''Definitions used for plots'''

import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use('fivethirtyeight')

def pie_plots(data,treatment_name, target_name):
    treatments = data[treatment_name].unique()
    explode = (0, 0.1) 
    fig, axes = plt.subplots(1, 4,figsize=(14,7))
    fig.set_facecolor('snow')

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

def _cat_gender(data):
    data['gender'] = 1
    data.loc[data['mens'] == 1, 'gender'] = 2
    return data

def plot_categorical(data):
    data = _cat_gender(data)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="whitegrid",rc=custom_params)
    fig, axes = plt.subplots(2,2,figsize=(15,11))
    fig.set_facecolor('snow')
    fig.suptitle('Categorical Features',y=0.94)


    sns.countplot(x="channel", hue="visit",  data=data, ax = axes[0,0],palette=sns.color_palette(['lightsteelblue','darkseagreen']))
    sns.countplot(x="gender", hue="visit",  data=data, ax = axes[0,1],palette=sns.color_palette(['lightsteelblue','darkseagreen']))
    axes[0,1].set_xticklabels(['women','men'])

    sns.countplot(x="zip_code", hue="visit",  data=data, ax = axes[1,0],palette=sns.color_palette(['lightsteelblue','darkseagreen']))
    sns.countplot(x="newbie", hue="visit",  data=data, ax = axes[1,1],palette=sns.color_palette(['lightsteelblue','darkseagreen']))
    axes[1,1].set_xticklabels(['no newbie','newbie'])

def plot_continuous(data):
    x = data[data['visit'] == 0]['recency']
    y = data[data['visit'] == 1]['recency']

    xx = data[data['visit'] == 0]['history']
    yy = data[data['visit'] == 1]['history']

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    fig.set_facecolor('snow')
    fig.suptitle('Continuous Features')

    axes[0].hist((x,y), histtype='bar', stacked=True, color=('lightsteelblue','darkseagreen'), label = ('no visit','visit'))
    axes[0].set_title('recency')
    axes[0].legend()
    axes[1].hist((xx,yy), histtype='bar', stacked=True, color=('lightsteelblue','darkseagreen'), label = ('no visit','visit'))
    axes[1].set_title('history')
    axes[1].legend()

    plt.show()
