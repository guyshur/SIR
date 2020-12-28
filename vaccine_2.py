# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

N = np.array([9212000]) # The populations
vaccine_gap = 28
v1 = [[0 for i in range(vaccine_gap)] for j in range(vaccine_gap)]
v2 = [0 for i in N]
y = [27628] # number of Infected.
z = [350000] # number of Recovered
x = [N[i] - y[i] - z[i] for i in range(len(N))]# number of Susceptible

d = [0] # number of Deceased
seger = False # boolean that represents a seger/lockdown
vaccines_remaining = 5 * 10 ** 6 # Number of vaccines available
vaccines_per_day = 50000 # Limit of daily vaccines
vaccine_efficency_1 = 0.5 # efficiency of the forst vaccine from 0 to 1
vaccine_efficency_2 = 0.9 # efficiency of the second vaccine from 0 to 1
yearly_birth_rate = 20/10000 # rate of birth in israel per year
yearly_death_rate = 5/10000 # rate of death in israel per year
daily_birth_rate = yearly_birth_rate / 365 # rate of birth in israel per day
daily_death_rate = yearly_death_rate / 365 # rate of death in israel per day
corona_deaths = 0 # number of people dead because of the virus during the simulation
time_between_vaccines = 14 # number of days between first and second dose of vaccines
# vectors of data generated during the simulation
taxis = []
xaxis = []
yaxis = []
vaxis = []
daxis = []
zaxis = []

seger_plot_color_change = [0] # this vector stores days when a lockdown starts or ends. it is used only to paint the graph in grey when there is a lockdown
beta = base_beta = 0.1 # beta is the S -> I transmission rate. base beta is the transmission rate when there is no lockdown
betas = [
    [0.1/N]
]
vaccine_infected = [0 for i in N]
lockdown_beta = [betas * 0.25 # lockdown_beta is the transmission rate when there is a lockdown
transition_time = 7 # the number of days it takes base_beta to drop to lockdown_beta
steps = (base_beta - lockdown_beta) / transition_time # the daily rate of decreasing beta when a lockdown starts
gamma = 0.075 - 0.00075 # recovery rate (I -> R)
lethality = 0.00075 # Corona death rate (I -> D)
dt = 0.001 # size of each step in the simulation = 1/1000 day
daily_infected = 0 # how many got infected in one day (1000 * dt)
t = 0 # the day
cnt = 0 # number of steps (same as dt)
days = 730 # how many days to run the simulation
while t < days: # the main loop
    if cnt % 1000 == 0: # If one day has passed, add the data to the graph
        taxis.append(t)
        xaxis.append(np.sum(x))
        yaxis.append(np.sum(y))
        zaxis.append(np.sum(z))
        vaxis.append(np.sum(v1))
        daxis.append(np.sum(d))

    # The number of people vaccinated in t time.
    # It equals min(vaccines_per_day, vaccines_remaining) unless floor(t) is a multiple of 7 (meaning shabbat, then it equals 0)
    vaccinations = min(vaccines_per_day, vaccines_remaining) if int(t) % 7 != 0 else 0

    # lower the amount of remaining vaccines
    vaccines_remaining -= vaccinations*dt


    for i in range(len(v1)):
        print(v1[i].T, y)
        infected = betas.dot(v1[i] * y * (1-vaccine_efficency_1))
        print(np.shape(v1[i]))
        print(np.shape(infected))
        v1[i] -= infected * dt
        vaccine_infected[i] = np.sum(infected)

    vaccintated = np.array([vaccinations])
    # dV / dt = number of people vaccinated - number of vaccinated that are infected
    dV1dt = vaccintated - vaccine_infected

    # dS / dt = - number of susceptibles that got infected - number of susceptibles that got vaccinated

    dSdt = -(betas.dot(x*y)) - vaccintated
    # dR / dt = number of recovered
    dRdt = (gamma * y)

    # dD / dt = number of deceased
    dDdt = y*lethality

    # sum up the total number of dead, to test validity of vaccine strategies in the future
    corona_deaths += y*lethality*dt

    # sum up number of infected per day, to decide if to start or end a lockdown
    daily_infected += (betas * x * y) * dt + (betas * v1 * y * (1-vaccine_efficency_1)) * dt

    x = x + dSdt * dt
    print(x)
    z = z + dRdt * dt
    # v1 = v1 + dV1dt * dt
    d = d + dDdt * dt
    print(N,z,x,np.sum(v1,axis=0),d)
    y = N - z - x - np.sum(v1, axis=0) - d
    print(y)
    t = t + dt
    cnt += 1
    exit(0)
    # Government response (decide on lockdown)
    if cnt % (1 / dt) == 0: # if dt is a multiple of 1 / dt (meaning one day has passed)



        if daily_infected > 4000:
            if seger is False:
                seger = True  # start a lockdown, next day start decreasing beta
                seger_plot_color_change.append(t) # this is only for painting the graph background color
            else: # keep the lockdown
                beta = min(lockdown_beta, beta - steps)  # beta is decreased unless it reached the minimum
        else:
            if seger is True:
                if daily_infected < 500:
                    seger = False  # end the seger, return beta to normal tomorrow
                    seger_plot_color_change.append(t) # this is only for painting the graph background color
            else:
                beta = base_beta  # return beta to normal value
        daily_infected = 0 # reset daily infected


# plot the data
plt.title("SIRVD MODEL")
plt.plot(taxis, xaxis, color=(0, 1, 0), linewidth=3.0, label='S')
plt.plot(taxis, yaxis, color=(1, 0, 0), linewidth=3.0, label='I')
plt.plot(taxis, zaxis, color=(0, 0, 1), linewidth=3.0, label='R')
plt.plot(taxis, vaxis, color=(0, 1, 1), linewidth=3.0, label='V')
plt.plot(taxis, daxis, color=(0, 0, 0), linewidth=1.0, label='D')
color = 'white'
for i in range(len(seger_plot_color_change) -1):
    plt.axvspan(seger_plot_color_change[i], seger_plot_color_change[i+1], facecolor=color, alpha=0.1)
    color = 'black' if color == 'white' else 'white'

plt.xlim(0, days)
plt.legend()
plt.xlabel('DAY')
plt.grid(True)
plt.show()
print("number of infected after {} days:".format(days), int(y)) # number of infected at the end of the simulation
print("number of dead from corona over {} days:".format(days), int(corona_deaths)) # number of dead (from corona) at the end of simulation