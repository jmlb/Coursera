# Uses python3
import sys

comp_speed = 'fast'

input = sys.stdin.read()
input_list = input.split('\n')
n = int(input_list[0])
a = [int(x) for x in input_list[1].split()]

assert(len(a) == n)

if comp_speed == 'slow':
    result = 0
    for i in range(0, n):
        for j in range(i+1, n):
            if a[i]*a[j] > result:
                result = a[i]*a[j]
    print(result)

if comp_speed == 'fast':
    max_index1 = -1;

    for i in range(0, n):
        if ((max_index1 == -1) or (a[i] > a[max_index1])):
            max_index1 = i;
    max_index2 = -1;
    for j in range(0, n):
        if ((j != max_index1) and ((max_index2 == -1) or (a[j] > a[max_index2]))):
            max_index2 = j;
    result = a[max_index1] * a[max_index2]
    print(result)