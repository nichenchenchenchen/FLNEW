import numpy as np
import matplotlib.pyplot as plt

class dynamicpainting():
    def __init__(self,rounds,figure):
        self.figure = figure
        plt.figure(figure)
        plt.axis([0,rounds + 10,0,1])
        plt.ion()
        self.xs = [0,0]
        self.ys = [0,0]
    def addData(self,y):
        #plt.figure(self.figure)
        self.xs[0],self.ys[0] = self.xs[1],self.ys[1]
        self.xs[1] += 1
        self.ys[1] = y
        plt.plot(self.xs,self.ys)
        plt.pause(0.1)


from matplotlib.animation import FuncAnimation   #导入负责绘制动画的接口
#其中需要输入一个更新数据的函数来为fig提供新的绘图信息
plt.figure(1)
plt.axis([0,10,0,1])
plt.ion()

fig, ax = plt.subplots()
x, y= [], []
line, = plt.plot([], [], '.-',color='orange')
nums = 50   #需要的帧数
plt.pause(5)
'''
def init():
    ax.set_xlim(-5, 60)
    ax.set_ylim(-3, 3)
    return line

def update(step):
    if len(x)>=nums:       #通过控制帧数来避免不断的绘图
        return line
    x.append(step)
    y.append(np.cos(step/3)+np.sin(step**2))    #计算y
    line.set_data(x, y)
    return line

ani = FuncAnimation(fig, update, frames=nums,     #nums输入到frames后会使用range(nums)得到一系列step输入到update中去
                    init_func=init,interval=20)
plt.show()

'''
