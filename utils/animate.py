import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from matplotlib import style


style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("../sampleText.txt","r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    zar = []
    for eachLine in dataArray:
        if len(eachLine)>2:
            x,y,z = eachLine.split(',')
            xar.append(int(x))
            yar.append(int(y))
            zar.append(int(z))

        ax1.clear()
        ax1.plot(xar,yar)
        ax1.plot(xar,zar)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.ylim((0, 10))
plt.xlim( (0,20) )
plt.show()


