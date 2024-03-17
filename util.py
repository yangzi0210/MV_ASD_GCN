def str2float(arr):
    for i in range(len(arr)):
        arr[i] = float(arr[i])
    return arr


def formatOutput(output):
    return "{:.8f}".format(output)


def meanOfArr(arr):
    return sum(arr) / len(arr)
