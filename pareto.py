import pandas as pd
import seaborn as sns
import pickle

import matplotlib.pyplot as plt

with open('loss.pickle', 'rb') as handle:
    loss_dict = pickle.load(handle)

n = len(loss_dict["img_name"])
x = [i+1 for i in range(n)]
imgs = loss_dict["img_name"]
front_error = loss_dict['degree_error_f']
right_error = loss_dict['degree_error_r']
up_error = loss_dict['degree_error_u']

def pareto_plot(df, x=None, y=None, title=None, show_pct_y=False, pct_format='{0:.0%}'):
    xlabel = x
    ylabel = y
    #tmp = df.sort_values(y, ascending=False)
    tmp = df
    x = tmp[x].values
    y = tmp[y].values
    weights = y / y.sum()
    cumsum = weights.cumsum()
    
    fig, ax1 = plt.subplots()
    ax1.bar(x, y)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2 = ax1.twinx()
    ax2.plot(x, cumsum, '-ro', alpha=0.5)
    ax2.set_ylabel('', color='r')
    ax2.tick_params('y', colors='r')
    
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    # hide y-labels on right side
    if not show_pct_y:
        ax2.set_yticks([])
    
    formatted_weights = [pct_format.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], cumsum[i]), fontweight='heavy')    
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    plt.draw()
    plt.savefig("up_error.png",dpi=fig.dpi)

dic = {}
dic['0-2'] = 0
dic['2-4'] = 0
dic['4-6'] = 0
dic['6-8'] = 0
dic['8-10'] = 0
dic['>10'] = 0


for err in up_error:
	if 0<err<=2:
		dic['0-2'] += 1
	elif 2<err<=4:
		dic['2-4'] += 1
	elif 4<err<=6:
		dic['4-6'] += 1
	elif 6<err<=8:
		dic['6-8'] += 1
	elif 8<err<=10:
		dic['8-10'] += 1
	else:
		dic['>10'] += 1


df = pd.DataFrame({
    'error_intervals': ['0-2', '2-4', '4-6','6-8','8-10','>10'],
    'frequency': [dic['0-2'],  dic['2-4'],  dic['4-6'],  dic['6-8'],   dic['8-10'], dic['>10']]
})


pareto_plot(df, x='error_intervals', y='frequency', title='Up Error')

