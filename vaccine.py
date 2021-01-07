# coding: utf-8

# Defining population sizes
import matplotlib.pyplot as plt
import numpy as np
Ntotal = 9207817.0
children = 4608434
high_risk = 1093072
other = Ntotal - children - high_risk
days_in_lockdown = 0
N = np.array([other,children,high_risk]) # The populations
groups = len(N) # number of different groups
vaccine_gap = 28 # time until second dose of vaccine (in days)
v1 = np.array([[0.0 for i in range(vaccine_gap)] for j in N]) # the V1 compartment matrix
v2 = np.array([0.0 for i in N]) # the V2 compartment vector
y = np.array([16941.0, 22266.0,5281.0]) # number of Infected.
z = np.array([142250.0,186963.0,44346.0]) # number of Recovered
vaccine_profile = np.array([0.5,0.3,0.2]) # The priority of which group to vaccinate (Note that 0 means the group will never be vaccinated, even if there are leftover vaccines)
assert(np.sum(vaccine_profile) == 1)
second_vaccine_priority = 0.5
assert(1 >= second_vaccine_priority >= 0)
x = N - y - z # number of Susceptibles

d = np.array([0.0,0.0,0.0]) # number of Deceased
seger = False # boolean that represents a seger/lockdown

vaccines_per_day = 50000 # Limit of daily vaccines

vaccine_efficency_1 = 0.52 # efficiency of the forst vaccine from 0 to 1
vaccine_efficency_2 = 0.95 # efficiency of the second vaccine from 0 to 1
immunity_loss_rate = 1/(30*6)
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
f2 = np.sqrt(4.7)
f1 = np.sqrt(6)
seger_plot_color_change = [0] # this vector stores days when a lockdown starts or ends. it is used only to paint the graph in grey when there is a lockdown
normal = [
    [0.09,0.09,0.09],
    [0.09,0.09,0.09],
    [0.09,0.09,0.09]
    ]
shedders = [
    [0.09,0.09,0.09],
    [0.09,0.09,0.09],
    [0.12,0.12,0.12]
]
spreaders = [
    [0.09,0.09,0.03],
    [0.06,0.06,0.073],
    [0.03,0.073,0.03]
]
bbb = [
    [0.08,0.10,0.09],
    [0.07,0.11,0.10],
    [0.06,0.10,0.11]
]
data_based_beta = 0.03
data_based = [
    [data_based_beta*f2*f2,data_based_beta*f1*f2,data_based_beta*f2],
    [data_based_beta*f2*f1,data_based_beta*f1*f1,data_based_beta*f1],
    [data_based_beta*f2,data_based_beta*f1,data_based_beta]
]
betas = np.array(data_based)/np.sum(Ntotal) # S -> I transmission matrix, betas at i,j = transmission from group j to group i
base_betas = np.copy(betas) # to be used when not in lockdown
vaccine_infected = np.array([0 for i in N])
lockdown_beta = betas * 0.25 # lockdown_beta is the transmission rate when there is a lockdown
transition_time = 7 # the number of days it takes base_beta to drop to lockdown_beta
steps_left = 7
steps = (betas - lockdown_beta) / transition_time # the daily rate of decreasing beta when a lockdown starts
gamma = np.array([0.075*0.9938,0.075*0.99970,0.075*0.926]) # recovery rate (I -> R)
lethality = np.array([0.075]*3) - gamma # Corona death rate (I -> D)
dt = 0.001 # size of each step in the simulation = 1/1000 day
daily_infected = 0 # how many got infected in one day (1000 * dt)
t = 0 # the day
cnt = 1 # number of steps (same as dt)
days = 365 # how many days to run the simulation


vaccines_remaining = 5 * 10 ** 6 # Number of vaccines available
while t < days: # the main loop
    if cnt % 100 == 0: # If 1/10th day has passed, add the data to the graph
        taxis.append(t)
        xaxis.append(np.sum(x))
        yaxis.append(np.sum(y))
        zaxis.append(np.sum(z))
        v1axis.append(np.sum(v1))
        v2axis.append(np.sum(v2))
        daxis.append(np.sum(d))
    v2_infected = (betas*(v2.reshape(groups, 1).dot(y.reshape(1, groups)))*(1-vaccine_efficency_2))*dt # How many people from V2 got infected
    daily_infected += np.sum(v2_infected)
    v2 -= np.sum(v2_infected,axis=1)
    # The number of people vaccinated in t time.
    # It equals min(vaccines_per_day, vaccines_remaining) unless floor(t) is a multiple of 7 (meaning shabbat, then it equals 0)
    vaccinations = min(vaccines_per_day, vaccines_remaining)*dt if int(t+1) % 7 != 0 else 0
    # vaccines_remaining -= vaccinations * dt
    ready_for_second_vaccine = np.copy(v1[:,-1])
    total_v2_this_iteration = []
    second_phase_vaccines = vaccinations * second_vaccine_priority
    first_phase_vaccines = vaccinations - second_phase_vaccines
    for i in range(len(ready_for_second_vaccine)):
        allocated_vaccines = min(vaccine_profile[i]*second_phase_vaccines,ready_for_second_vaccine[i])
        total_v2_this_iteration.append(allocated_vaccines)
    total_v2_this_iteration = np.array(total_v2_this_iteration)
    vaccines_remaining -= np.sum(total_v2_this_iteration)
    leftover_v2 = second_phase_vaccines - np.sum(total_v2_this_iteration)
    first_phase_vaccines += leftover_v2


    v1[:, -1] -= total_v2_this_iteration
    v2 += total_v2_this_iteration

    total_v1_this_iteration = []
    for i in range(groups):
        allocated_vaccines = min(vaccine_profile[i]*first_phase_vaccines,x[i])
        total_v1_this_iteration.append(allocated_vaccines)
    total_v1_this_iteration = np.array(total_v1_this_iteration)
    vaccines_remaining -= np.sum(total_v1_this_iteration)


    v1[:,0] += total_v1_this_iteration

    for i in range(len(v1)):
        bet = betas[i]
        bet = bet * y
        infected = np.zeros(vaccine_gap)
        for j in range(groups):
            infected += v1[i] * bet[j] * (1-vaccine_efficency_1)

        v1[i] -= infected * dt
        daily_infected += np.sum(infected*dt, dtype=float)
    # vaccines_remaining = vaccines_remaining - np.sum(planned_vaccinations_1) - np.sum(total_v2_this_iteration)
    dSdt = np.sum( -betas*(x.reshape(groups, 1).dot(y.reshape(1, groups))),axis=1)
    daily_infected -= np.sum(dSdt*dt)

    dSdt += z*immunity_loss_rate
    dRdt = (gamma * y) - z*immunity_loss_rate
    dDdt = y*lethality

    # sum up the total number of dead, to test validity of vaccine strategies in the future
    corona_deaths += np.sum(y*lethality*dt)

    x = x + dSdt * dt
    x -= total_v1_this_iteration
    z = z + dRdt * dt
    d = d + dDdt * dt
    y = N - z - x - np.sum(v1, axis=1) - v2 - d
    t = t + dt
    cnt += 1

    if cnt % (1 / dt) == 0: # if dt is a multiple of 1 / dt (meaning one day has passed)
        if seger:
            days_in_lockdown += 1
        # Shift V1 matrix one day (reset day 1, shift days 1 to 26 one spot to the right, and add day 27 to day 28+)
        new = np.zeros(v1.shape)
        new[:, 1:-1] = v1[:, :-2]
        new[:, -1] = np.sum(v1[:, -2:], axis=1)
        v1 = new

        # Government response (decide on lockdown)
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

plt.xlim(0, days)
plt.legend()
plt.xlabel('DAY')
plt.grid(True)
plt.savefig('base_case_vaccines.png')
plt.show()
print("number of infected after {} days:".format(days), int(np.sum(y))) # number of infected at the end of the simulation
print("number of dead from corona over {} days:".format(days), int(corona_deaths)) # number of dead (from corona) at the end of simulation
print("days in lockdown:",days_in_lockdown)