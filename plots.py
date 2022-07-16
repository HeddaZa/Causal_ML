"""Definitions used for plots"""

import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('fivethirtyeight')


def pie_plots(data, treatment_name, target_name):
    treatments = data[treatment_name].unique()
    explode = (0, 0.1)
    fig, axes = plt.subplots(1, 4, figsize=(14, 7))
    fig.set_facecolor("snow")

    axes[0].pie(
        data.groupby(target_name)[treatment_name].count(),
        textprops={"fontsize": 14},
        colors=["lightsteelblue", "darkseagreen"],
        shadow=True,
        labels=["no visit", "visit"],
        explode=explode,
        autopct="%1.0f%%",
    )
    axes[0].set_title("All treatments")
    for i, treatment in enumerate(treatments):
        axes[i + 1].pie(
            data[data[treatment_name] == treatment]
            .groupby(target_name)["recency"]
            .count(),
            textprops={"fontsize": 14},
            colors=["lightsteelblue", "darkseagreen"],
            shadow=True,
            labels=["no visit", "visit"],
            explode=explode,
            autopct="%1.0f%%",
        )
        axes[i + 1].set_title(treatment)


def _cat_gender(data):
    data["gender"] = 1
    data.loc[data["mens"] == 1, "gender"] = 2
    return data

def map_visit(series):
    map_visit = {0:'no visit', 1: 'visit'}
    return series.map(map_visit)


def plot_categorical(data):
    data = _cat_gender(data)
    data['visit'] = map_visit(data['visit'])
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="whitegrid", rc=custom_params)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.set_facecolor("snow")
    fig.suptitle("Categorical Features", y=0.94)

    sns.countplot(
        x="channel",
        hue="visit",
        data=data,
        ax=axes[0, 0],
        palette=sns.color_palette(["lightsteelblue", "darkseagreen"]),
    )
    sns.countplot(
        x="gender",
        hue="visit",
        data=data,
        ax=axes[0, 1],
        palette=sns.color_palette(["lightsteelblue", "darkseagreen"]),
    )
    axes[0, 1].set_xticklabels(["women", "men"])

    sns.countplot(
        x="zip_code",
        hue="visit",
        data=data,
        ax=axes[1, 0],
        palette=sns.color_palette(["lightsteelblue", "darkseagreen"]),
    )
    sns.countplot(
        x="newbie",
        hue="visit",
        data=data,
        ax=axes[1, 1],
        palette=sns.color_palette(["lightsteelblue", "darkseagreen"]),
    )
    axes[1, 1].set_xticklabels(["no newbie", "newbie"])


def plot_continuous(data):
    x = data[data["visit"] == 'no visit']["recency"]
    y = data[data["visit"] == 'visit']["recency"]

    xx = data[data["visit"] == 'no visit']["history"]
    yy = data[data["visit"] == 'visit']["history"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.set_facecolor("snow")
    fig.suptitle("Continuous Features")

    axes[0].hist(
        (x, y),
        histtype="bar",
        stacked=True,
        color=("lightsteelblue", "darkseagreen"),
        label=("no visit", "visit"),
    )
    axes[0].set_title("recency")
    axes[0].legend()
    axes[0].set_ylabel('count')
    axes[1].hist(
        (xx, yy),
        histtype="bar",
        stacked=True,
        color=("lightsteelblue", "darkseagreen"),
        label=("no visit", "visit"),
    )
    axes[1].set_title("history")
    axes[1].legend()
    axes[1].set_ylabel('dollars spent')

    plt.show()


def plot_ERUPT(S_dict, T_dict, cST_dict):

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 7))
    fig.set_facecolor("snow")

    ax[0].boxplot(
        [S_dict["logreg"][2], S_dict["randfor"][2], S_dict["xgb"][2]], notch=True
    )
    ax[0].set_xticklabels(["log. reg.", "rand. for.", "xgb"])
    ax[0].plot(
        1,
        S_dict["logreg"][0],
        "o",
        color="plum",
        label="ERUPT value models",
        mec="black",
        mew=0.4,
        ms=11,
    )
    ax[0].plot(2, S_dict["randfor"][0], "o", color="plum", mec="black", mew=0.4, ms=11)
    ax[0].plot(3, S_dict["xgb"][0], "o", color="plum", mec="black", mew=0.4, ms=11)
    ax[0].set_title("S-learner", fontsize=20)
    ax[0].axhline(
        S_dict["logreg"][1],
        c="mediumseagreen",
        label="mean of visits in test set",
        lw=3,
    )

    ax[1].boxplot(
        [T_dict["logreg"][2], T_dict["randfor"][2], T_dict["xgb"][2]], notch=True
    )
    ax[1].set_xticklabels(["log. reg.", "rand. for.", "xgb"])
    ax[1].plot(1, T_dict["logreg"][0], "o", color="plum", mec="black", mew=0.4, ms=11)
    ax[1].plot(2, T_dict["randfor"][0], "o", color="plum", mec="black", mew=0.4, ms=11)
    ax[1].plot(3, T_dict["xgb"][0], "o", color="plum", mec="black", mew=0.4, ms=11)
    ax[1].set_title("S-learner", fontsize=20)
    ax[1].axhline(T_dict["logreg"][1], c="mediumseagreen", lw=3)

    ax[1].set_title("T-learner", fontsize=20)

    ax[2].boxplot(cST_dict["xgb"][2], notch=True)

    ax[2].axhline(cST_dict["xgb"][1], c="mediumseagreen", lw=3)
    ax[2].plot(1, cST_dict["xgb"][0], "o", color="plum", mec="black", mew=0.4, ms=11)
    ax[2].set_title("corr ST_Learner", fontsize=20)
    ax[2].set_xticklabels(["xgb"])
    fig.legend(
        loc="upper center", bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True, shadow=True
    )

    plt.subplots_adjust(hspace=0.1)

    fig.text(0.5, 0.07, " ", ha="center", fontsize="x-large")
    fig.text(
        0.07, 0.5, "ERUPT values", va="center", rotation="vertical", fontsize="x-large"
    )

    plt.plot()
