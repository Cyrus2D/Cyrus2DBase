import matplotlib
import matplotlib.pyplot as plt

# Read the file
file_path = 'allEndReards.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

rewards = [float(line.strip()) for line in lines]

averages = [sum(rewards[i:i+100]) / 100 for i in range(0, len(rewards), 100)]

# Plot the averages
plt.plot(averages)
plt.xlabel('Group')
plt.ylabel('Average Reward')
plt.title('Average Reward per Group')

# Save the plot to a file
plt.savefig('plot.png')
