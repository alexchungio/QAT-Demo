import numpy as np
from scipy.optimize import leastsq


# 定义圆的方程
def circle_func(params, x, y):
    x0, y0, r = params
    return (x - x0) ** 2 + (y - y0) ** 2 - r ** 2


if __name__ == "__main__":
    # 用于拟合的数据点
    data_x = np.array([1, 2, 3, 4, 5])
    data_y = np.array([1, 2, 3, 4, 5])

    # 初始参数：圆心坐标(x0, y0)和半径r
    initial_params = np.array([0.0, 0.0, 1.0])

    # 使用leastsq进行拟合
    params, cov = leastsq(circle_func, initial_params, args=(data_x, data_y))

    # 输出拟合参数
    print(params)

    # 绘制数据点和拟合的圆
    import matplotlib.pyplot as plt

    # 假设数据点在一个单位圆内
    x = np.linspace(-1, 1, 100)
    y = np.sqrt(1 - (x - params[0]) ** 2) + params[1]

    plt.scatter(data_x, data_y)
    plt.plot(x, y, color='red')
    plt.axis('equal')
    plt.show()