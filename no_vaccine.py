# coding: utf-8
import math
import matplotlib.pyplot as plt
# N1 = arabs, N2 = hasidic, N3 = other
N = 9212000
N1 = N*0.209
N2 = 1125000
N3 = N - N1 - N2
y = old_y = 27628
z = old_z = 350000
x = N-y-z
seger = False
vaccines = 5 * 10 ** 6
vaccines_per_day = 50000

taxis=[ ]
xaxis=[ ]
yaxis=[ ]
zaxis=[ ]
seger_arr=[]
beta=base_beta = 0.1
gamma=0.075
dt=0.001
daily_infected = 0
vaccine_rate = vaccines_per_day * dt
t = 0
cnt=0
while t<730:
    if cnt%1000==0:
        taxis.append(t)
        xaxis.append(x)
        yaxis.append(y)
        zaxis.append(z)

    dSdt1 = (-beta*x*y)/N
    dRdt1 = gamma*y

    x2 = x + (dSdt1 * dt)
    z2 = z + (dRdt1 * dt)
    y2 = N - x - z
    dSdt2 = (-beta * x2 * y2) / N
    dRdt2 = gamma * y2
    daily_infected = daily_infected - ((dSdt1 + dSdt2) * dt)/2
    x = x + ((dSdt1 + dSdt2) * dt)/2
    z = z + ((dRdt1 + dRdt2) * dt)/2
    y = N - z - x
    t = t + dt
    cnt += 1

    # Government response

    if cnt % (1 / dt) == 0:

        if seger == True:
            for i in range(1):
                seger_arr.append(10**6)
        else:
            for i in range(1):
                seger_arr.append(0)

        if daily_infected > 5000:
            if seger == False:
                seger = True # next week decrease beta
            else:
                beta = base_beta * 0.5 # tighten seger
        else:
            if seger == True:
                if daily_infected < 500:
                    seger = False # next week start increasing beta
            else:
                beta = min(base_beta, base_beta) # relax seger
        old_y = y
        old_z = z
        print(daily_infected)
        daily_infected = 0
        
extra_days = len(taxis) - len(seger_arr)
for i in range(extra_days):
    seger_arr.append(0)

plt.title("SIR MODEL")
plt.plot(taxis,xaxis, color=(0,1,0), linewidth=3.0, label='S')
plt.plot(taxis,yaxis, color=(1,0,0), linewidth=3.0, label='I')
plt.plot(taxis,zaxis, color=(0,0,1), linewidth=3.0, label='R')
plt.plot(taxis,seger_arr, alpha=0.3, color=(0,0,0), linewidth=5.0, label='seger')

plt.xlim(0,730)
plt.legend()
plt.xlabel('DAY')
plt.grid(True)
plt.show()
print("number of dead:",z*0.01)