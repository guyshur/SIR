# coding: utf-8

# Defining population sizes
import matplotlib.pyplot as plt
import numpy as np
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
f2 = np.sqrt(4.7)
f1 = np.sqrt(6)
old_data_based = [
    [data_based_beta*f2*f2,data_based_beta*f1*f2,data_based_beta*f2],
    [data_based_beta*f2*f1,data_based_beta*f1*f1,data_based_beta*f1],
    [data_based_beta*f2,data_based_beta*f1,data_based_beta]
]
data_based = [
  [0.0893000000000000,0.100896977159873,0.0411910184384897],
   [0.100896977159873,0.114000000000000,0.0465403051128804],
   [0.0411910184384897,0.0465403051128804,0.0190000000000000]
] # S -> I transmission matrix, betas at i,j = transmission from group j to group i
default_vaccine_profile = np.array([0.1,0.9,0.0])

def simulate(vaccine_profile = None, beta = None, second_vaccine_priority = 0.5, vaccines = True):
    vaccine_profile = default_vaccine_profile if vaccine_profile is None else np.array(vaccine_profile)
    beta = data_based if beta is None else beta
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
    assert(0.9999 <= np.sum(vaccine_profile) <= 1.00001)
    assert(1 >= second_vaccine_priority >= 0)
    x = N - y - z # number of Susceptibles

    d = np.array([0.0,0.0,0.0]) # number of Deceased
    seger = False # boolean that represents a seger/lockdown

    vaccines_per_day = 50000 # Limit of daily vaccines

    vaccine_efficency_1 = 0.52 # efficiency of the forst vaccine from 0 to 1
    vaccine_efficency_2 = 0.95 # efficiency of the second vaccine from 0 to 1
    immunity_loss_rate = 1/(30*6)
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


    betas = np.array(beta)/np.sum(Ntotal) # S -> I transmission matrix, betas at i,j = transmission from group j to group i
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


    vaccines_remaining = 5 * 10 ** 6 if vaccines else 0 # Number of vaccines available
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
    vaccine_profile_str = (','.join([str(item) for item in vaccine_profile]))
    v1_distribution = (','.join([str(round(item,3)) for item in list(np.sum(v1,axis=1)/N)]))
    v2_distribution = (','.join([str(round(item,3)) for item in list(v2/N)]))
    second_vaccine_priority = str(round(second_vaccine_priority,3))
    output = ','.join([vaccine_profile_str,second_vaccine_priority,str(int(corona_deaths)),str(int(np.sum(y))),str(days_in_lockdown),v1_distribution,v2_distribution])
    # plot the data
    plt.title("SIR MODEL")
    plt.plot(taxis, xaxis, color=(0, 1, 0), linewidth=3.0, label='S')
    plt.plot(taxis, yaxis, color=(1, 0, 0), linewidth=3.0, label='I')
    plt.plot(taxis, zaxis, color=(0, 0, 1), linewidth=3.0, label='R')
    plt.plot(taxis, v1axis, color=(0, 1, 1), linewidth=3.0, label='V1')
    plt.plot(taxis, v2axis, color=(1, 0, 1), linewidth=3.0, label='V2')
    #plt.plot(taxis, daxis, color=(0.0, 0.0, 0.0), linewidth=1.0, label='D', alpha=1)
    color = 'white'
    for i in range(len(seger_plot_color_change) -1):
        plt.axvspan(seger_plot_color_change[i], seger_plot_color_change[i+1], facecolor=color, alpha=0.1)
        color = 'black' if color == 'white' else 'white'

    plt.xlim(0, days)
    plt.legend()
    plt.xlabel('DAY')
    plt.grid(True)
    plt.savefig('last_result.png')
    plt.show()
    return int(corona_deaths), np.sum(y), days_in_lockdown, output


def best_profile_search():
    results = []
    best_profile_for_lockdown = None
    best_dose_priority_for_lockdown = 0
    best_lockdown = np.inf
    best_profile_for_deaths = None
    best_dose_priority_for_deaths = 0
    best_deaths = np.inf
    for i in range(11):
        for j in range(11):
            for k in range(11):
                    if i+j+k == 10:
                        profile = [i*0.1,j*0.1,k*0.1]
                        for l in range(7):
                            deaths,infected,lockdown,output = simulate(vaccine_profile=profile, beta=data_based
                                                                , vaccines=True,second_vaccine_priority=l*(1/6))
                            print(profile.__str__()+':',deaths,infected,lockdown)
                            results.append(output)
                            if deaths < best_deaths:
                                best_profile_for_deaths = profile
                                best_dose_priority_for_deaths = l*(1/6)
                                best_deaths = deaths
                            if lockdown < best_lockdown:
                                best_profile_for_lockdown = profile
                                best_lockdown = lockdown
                                best_dose_priority_for_lockdown = l*(1/6)
    with open('parameter_search_full_result.csv','w+') as fp:
        fp.write('priority_for_other,priority_for_children,priority_for_risk_group,priority_for_vaccine_2,deaths,infected_at_end,days_in_lockdown,v1_other/N(other),v1_children/N(children),v1_risk_group/N(risk_group),v2_other/N(other),v2_children/N(children),v2_risk_group/N(risk_group\n')
        fp.write('\n'.join(results))
    with open('parameter_search_best_result.txt','w+') as fp:
        fp.write('best death prevention profile - ' + best_profile_for_deaths.__str__() + ' second_dose_priority - ' + str(best_dose_priority_for_deaths) + ' result - ' + str(best_deaths) +
                 '\nbest lockdown prevention - ' + best_profile_for_lockdown.__str__() +  ' second_dose_priority - ' + str(best_dose_priority_for_lockdown) + ' result - ' +str(best_lockdown))


if __name__ == '__main__':
    print(simulate(vaccine_profile=default_vaccine_profile,beta=data_based,second_vaccine_priority=0,vaccines=True))