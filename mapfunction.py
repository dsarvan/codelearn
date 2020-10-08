#!/usr/bin/env python3

numbers = [1, 2, 3, 4, 5]

for i in range(0, len(numbers)):
    numbers[i] += 1

print(numbers)


numbers = [1, 2, 3, 4, 5]
result = []

for n in numbers:
    result.append(n+1)

print(result)


numbers = [1, 2, 3, 4, 5]
result = [n+1 for n in numbers]
print(result)


numbers = (1, 2, 3, 4, 5)
result = map(lambda x: x+1,  numbers)
print(result)
