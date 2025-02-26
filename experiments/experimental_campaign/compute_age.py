import numpy as np

candidates = ['Christian', 'Xenia', 'Mihailo', 'Marko', 'Eleonora', 'Bozidar']
status = ['expert', 'novice', 'expert', 'expert', 'novice', 'novice']
ages = [26, 24, 26, 25, 25, 24]
heights = [182, 168, 155, 192, 192, 195]

mean_height = np.mean(heights)
mean_age = np.mean(ages)
print(f"Mean height: {mean_height}")
print(f"Mean age: {mean_age}")

std_height = np.std(heights)
std_age = np.std(ages)
print(f"Standard deviation of height: {std_height}")
print(f"Standard deviation of age: {std_age}")