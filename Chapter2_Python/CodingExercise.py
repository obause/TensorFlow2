# type: ignore
import matplotlib.pyplot as plt
import numpy as np


def e_function(my_list):
    my_result = []

    for val in my_list:
        my_result.append(np.exp(val))

    return my_result


def e_function2(my_list):
    return [np.exp(val) for val in my_list]


def e_function3(my_list):
    return np.exp(my_list)


my_list = [1, 2, 3, 4, 5]
e_list = e_function(my_list)

print(e_function(my_list))
print(e_function2(my_list))
print(e_function3(my_list))

#          x        y
plt.plot(my_list, e_list, color="blue")
plt.xlabel("x")
plt.ylabel("e(x)")
plt.show()
