import matplotlib.pyplot as plt

Temperature=[
        24,23.7,23.2,23.3,
        23.1,23.3,24,25.4,
        27.8,29.4,30,29.4,
        29.7,30.6,31.3,30.7,
        29.8,28.8,28.2,27.9,
        27.4,27.1,26.4,26.6]
times=[i+1 for i in range(len(Temperature))]

plt.plot(times, Temperature, label='day temperature')
plt.legend() 
plt.xticks(times)
plt.show() 
