# coding: utf-8

# Defining population sizes
import matplotlib.pyplot as plt
import numpy as np
Ntotal = 9212000.0
children = 2.5*10**6
high_risk = 2*10**6
other = Ntotal - children - high_risk

N = np.array([other,children,high_risk]) # The populations
groups = len(N) # number of different groups
vaccine_gap = 28 # time until second dose of vaccine (in days)
v1 = np.array([[0.0 for i in range(vaccine_gap)] for j in N]) # the V1 compartment matrix
v2 = np.array([0.0 for i in N]) # the V2 compartment vector
y = np.array([8000.0, 16000.0,5000.0]) # number of Infected.
z = np.array([120000.0,220000.0,20000.0]) # number of Recovered
vaccine_profile = np.array([0.5,0.5,0.0]) # The priority of which group to vaccinate (Note that 0 means the group will never be vaccinated, even if there are leftover vaccines)
assert(np.sum(vaccine_profile) == 1)
x = N - y - z # number of Susceptibles

d = np.array([0.0,0.0,0.0]) # number of Deceased
seger = False # boolean that represents a seger/lockdown

vaccines_per_day = 50000 # Limit of daily vaccines

vaccine_efficency_1 = 0.5 # efficiency of the forst vaccine from 0 to 1
vaccine_efficency_2 = 0.9 # efficiency of the second vaccine from 0 to 1
yearly_birth_rate = 20/10000 # rate of birth in israel per year
yearly_death_rate = 5/10000 # rate of death in israel per year
daily_birth_rate = yearly_birth_rate / 365 # rate of birth in israel per day
daily_death_rate = yearly_death_rate / 365 # rate of death in israel per day
corona_deaths = 0 # number of people dead because of the virus during the simulation
# vectors of data generated during the simulation
taxis = []
xaxis = []
yaxis = []
v1axis = []
v2axis = []
daxis = []
zaxis = []

seger_plot_color_change = [0] # this vector stores days when a lockdown starts or ends. it is used only to paint the graph in grey when there is a lockdown
betas = np.array([
    [0.1,0.05,0.05],
    [0.15,0.2,0.025],
    [0.05,0.025,0.025]
])/np.sum(Ntotal) # S -> I transmission matrix, betas at i,j = transmission from group j to group i
base_betas = np.copy(betas) # to be used when not in lockdown
vaccine_infected = np.array([0 for i in N])
lockdown_beta = betas * 0.25 # lockdown_beta is the transmission rate when there is a lockdown
transition_time = 7 # the number of days it takes base_beta to drop to lockdown_beta
steps_left = 7
steps = (betas - lockdown_beta) / transition_time # the daily rate of decreasing beta when a lockdown starts
gamma = np.array([0.075,0.075*0.99,0.075*0.9]) # recovery rate (I -> R)
lethality = np.array([0.075]*3) - gamma # Corona death rate (I -> D)
dt = 0.001 # size of each step in the simulation = 1/1000 day
daily_infected = 0 # how many got infected in one day (1000 * dt)
t = 0 # the day
cnt = 1 # number of steps (same as dt)
days = 365 # how many days to run the simulation


vaccines_remaining = 5 * 10 ** 6 # Number of vaccines available
while t < days: # the main loop
    if cnt % 100 == 0: # If one day has passed, add the data to the graph
        taxis.append(t)
        xaxis.append(np.sum(x))
        yaxis.append(np.sum(y))
        zaxis.append(np.sum(z))
        v1axis.append(np.sum(v1))
        v2axis.append(np.sum(v2))
        daxis.append(np.sum(d))

    v2_infected = betas*(v2.reshape(groups, 1).dot(y.reshape(1, groups)))*(1-vaccine_efficency_2)
    daily_infected += np.sum(v2_infected) * dt
    # The number of people vaccinated in t time.
    # It equals min(vaccines_per_day, vaccines_remaining) unless floor(t) is a multiple of 7 (meaning shabbat, then it equals 0)
    vaccinations = min(vaccines_per_day, vaccines_remaining) if int(t+1) % 7 != 0 else 0
    vaccines_remaining -= vaccinations * dt
    ready_for_second_vaccine = np.copy(v1[:,-1])
    vaccinations_2 = min(np.sum(ready_for_second_vaccine), vaccinations/1.5)
    vaccinations_1 = vaccinations - vaccinations_2
    if vaccinations_2 > 0:
        profile = ready_for_second_vaccine * (vaccinations_2 / np.sum(ready_for_second_vaccine))
        v2 += profile * dt
        v1[:,-1] = v1[:,-1] - profile * dt


    planned_vaccinations_1 = vaccinations_1 * vaccine_profile
    for i in range(groups):
        if planned_vaccinations_1[i] > x[i]:
            leftovers = vaccine_profile[i]
            vaccine_profile[i] = 0
            planned_vaccinations_1[i] = 0
            for j in range(groups):
                if planned_vaccinations_1[j] > 0 and planned_vaccinations_1[j] + leftovers < x[i]:
                    vaccine_profile[j] += leftovers
                    break


    v1[:,0] += vaccinations_1*vaccine_profile*dt
    for i in range(len(v1)):
        bet = betas[i]
        bet = bet * y
        infected = np.zeros(vaccine_gap)
        for j in range(groups):
            infected += v1[i] * bet[j] * (1-vaccine_efficency_1)

        v1[i] -= infected * dt
        daily_infected += np.sum(infected*dt, dtype=float)

        vaccine_infected[i] = np.sum(infected,dtype=float)

    dSdt = np.sum( -betas*(x.reshape(groups, 1).dot(y.reshape(1, groups))),axis=1)
    daily_infected -= np.sum(dSdt*dt)
    dSdt -= planned_vaccinations_1
    dRdt = (gamma * y)
    dDdt = y*lethality

    # sum up the total number of dead, to test validity of vaccine strategies in the future
    corona_deaths += np.sum(y*lethality*dt)

    x = x + dSdt * dt
    z = z + dRdt * dt
    d = d + dDdt * dt
    y = N - z - x - np.sum(v1, axis=1) - v2 - d
    t = t + dt
    cnt += 1
    # Government response (decide on lockdown)
    if cnt % (1 / dt) == 0: # if dt is a multiple of 1 / dt (meaning one day has passed)
        new = np.zeros(v1.shape)
        new[:, 1:-1] = v1[:, :-2]
        new[:, -1] = np.sum(v1[:, -2:], axis=1)
        v1 = new
        if seger is True:
            if daily_infected > 500:
                if steps_left > 0:  # beta is decreased unless it reached the minimum
                    steps_left -= 1
                    betas -= steps
            else:
                seger = False
                seger_plot_color_change.append(t)  # this is only for painting the graph background color

        else:
            if daily_infected > 4000:
                seger = True  # start a lockdown, next day start decreasing beta
                seger_plot_color_change.append(t) # this is only for painting the graph background color
            else:
                if steps_left < 7:  # beta is decreased unless it reached the minimum
                    steps_left += 1
                    betas += steps

        daily_infected = 0 # reset daily infected


# plot the data
plt.title("SIRVD MODEL")
plt.plot(taxis, xaxis, color=(0, 1, 0), linewidth=3.0, label='S')
plt.plot(taxis, yaxis, color=(1, 0, 0), linewidth=3.0, label='I')
plt.plot(taxis, zaxis, color=(0, 0, 1), linewidth=3.0, label='R')
plt.plot(taxis, v1axis, color=(0, 1, 1), linewidth=3.0, label='V1')
plt.plot(taxis, v2axis, color=(1, 0, 1), linewidth=3.0, label='V2')
plt.plot(taxis, daxis, color=(0, 0, 0), linewidth=1.0, label='D')
color = 'white'
for i in range(len(seger_plot_color_change) -1):
    plt.axvspan(seger_plot_color_change[i], seger_plot_color_change[i+1], facecolor=color, alpha=0.1)
    color = 'black' if color == 'white' else 'white'

plt.xlim(0, 180)
plt.legend()
plt.xlabel('DAY')
plt.grid(True)
plt.show()
print("number of infected after {} days:".format(days), int(np.sum(y))) # number of infected at the end of the simulation
print("number of dead from corona over {} days:".format(days), int(corona_deaths)) # number of dead (from corona) at the end of simulation