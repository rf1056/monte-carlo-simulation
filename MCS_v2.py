import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

df = pd.read_excel('MCS.xlsx')

risks = df[df['Type'] == 'Risk']
opportunities = df[df['Type'] == 'Opportunity']

num_simulations = 10000

total_costs = []

#simulation

for _ in range(num_simulations):
    total_cost = 0

    #Risk Impact
    for _, row in risks.iterrows(): #left = mejor caso, right = peor caso
        left = min(row['Min'], row['Mode'], row['Max']) #mejor caso min valor
        mode = sorted([row['Min'], row['Mode'], row['Max']])[1]
        right = max(row['Min'], row['Mode'], row['Max']) #peor caso max valor
        if left == right:
            total_cost += left
        else:
            total_cost += np.random.triangular(left, mode, right)

    #Opportunity Impact
    for _, row in opportunities.iterrows(): 
        values = sorted([row['Min'],row['Max'],row['Mode']])
        left = values[0] #menor costo, mejor condicion
        mode = values[1] #intermedio
        right = min(values, key=abs) #mas cercano a 0, peor condicion, menos ahorro

        if left == right:
            total_cost += left
        else:
            total_cost += np.random.triangular(left, mode, right)


    total_costs.append(total_cost)


total_costs = np.array(total_costs)

mean_cost = np.mean(total_costs)
std_dev_cost = np.std(total_costs)
cost_5th_percentile = np.percentile(total_costs, 5)
cost_95th_percentile = np.percentile(total_costs, 95)
p50 = np.percentile(total_costs, 50)
p80 = np.percentile(total_costs, 80)


print(f"Mean total cost: £{mean_cost: .2f}")
print(f"Standard Deviation: £{std_dev_cost: .2f}")
print(f"5th percentile: £{cost_5th_percentile: .2f}")
print(f"95th percentile: £{cost_95th_percentile: .2f}")
print(f"p50 is £{p50: .2f} ")
print(f"p80 is £{p80: .2f}")


#grafico 1: Histograma (campana de gauss)
plt.figure(figsize=(10,6))

sns.histplot(total_costs, kde=True, bins=50, color='blue', stat="probability", linewidth=0)

formatter = FuncFormatter(lambda x, pos: f'£{x / 1000 : .0f}k')
plt.gca().xaxis.set_major_formatter(formatter)

plt.xticks(rotation=45,ha="right")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True,prune='both'))

plt.axvline(mean_cost, color='red',linestyle = '--', linewidth = 2, label = f'Media: £{mean_cost:,.0f}')
plt.axvline(cost_5th_percentile, color='green',linestyle = '--', linewidth = 2, label = f'5%: £{cost_5th_percentile:,.0f}')
plt.axvline(cost_95th_percentile, color='purple',linestyle = '--', linewidth = 2, label = f'95%: £{cost_95th_percentile:,.0f}')

plt.title('Distribucion de costos totales simulados - Montecarlo', fontsize=14)
plt.xlabel('Costo Total (£)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)

plt.show()

#grafico 2 linea acumulada
plt.figure(figsize=(10,6))

sorted_costs = np.sort(total_costs)
probabilities = np.linspace(0, 1, len(sorted_costs))

y_p50=np.interp(p50,sorted_costs,probabilities)
y_p80=np.interp(p80,sorted_costs,probabilities)

plt.plot(sorted_costs, probabilities, color='green', linewidth=2, label="CDF")

plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45,ha="right")

plt.plot([p50,p50],[0,y_p50], color="red",linestyle = '--', linewidth = 2, label = f'p50: £{p50:,.0f}')
plt.plot([p80,p80],[0,y_p80], color="blue",linestyle = '--', linewidth = 2, label = f'p80: £{p80:,.0f}')
plt.plot([6350000,p50],[y_p50,y_p50], color="red",linestyle = '--', linewidth = 2, label = f'p50: £{p50:,.0f}')
plt.plot([6350000,p80],[y_p80,y_p80], color="blue",linestyle = '--', linewidth = 2, label = f'p80: £{p80:,.0f}')

plt.title("Funcion de distribucion acumulada del costo total (CDF)",fontsize=16)
plt.xlabel("Costo total en miles de libras",fontsize=12)
plt.ylabel("Probabilidad Acumulada", fontsize=12)
plt.grid(True)
plt.show()