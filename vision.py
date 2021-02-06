import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class dynamicpainting():
    def __init__(self,rounds,subplot,title,initValue):
        ax = plt.subplot(2, 2, subplot)
        ax.set_title(title)
        self.subplot = subplot
        self.xs = [0,0]
        self.ys = [initValue,initValue]
    def addData(self,y):
        plt.subplot(2,2,self.subplot)
        self.xs[0],self.ys[0] = self.xs[1],self.ys[1]
        self.xs[1] += 1
        self.ys[1] = y
        plt.plot(self.xs,self.ys)
        plt.pause(0.2)


