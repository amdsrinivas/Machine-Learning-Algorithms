from matplotlib import pyplot as plt
import random
def distance(point1, point2):
    return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**(0.5)


# Generate random data.
data = []
while len(data)<=50:
    item = (random.randint(-10,10), random.randint(-10,10))
    if item not in data:
        data.append(item)




positives = []
negatives = []

# Define the positive examples according to (X^2 + Y^2 + 3X + 2Y <= 49)

for item  in data:
    if item[0]**2 + item[1]**2 + 3*item[0] + 2*item[1] <= 49:
        positives.append(item)
    else:
        negatives.append(item)

print(len(positives), len(negatives))

x = [ i[0] for i in positives]
y = [ i[1] for i in positives]

plt.scatter(x,y, color='orange')

x = [ i[0] for i in negatives]
y = [ i[1] for i in negatives]
plt.scatter(x,y,color='red')
#plt.show()

x = [ i[0] for i in positives]
y = [ i[1] for i in positives]

# Itertively extend the region based on the postive examples.
center = (sum(x)/len(x) , sum(y)/len(y))
r = 0
for point in positives:
    temp = distance(center, point)
    if temp > r:
        r = temp

print(r)
circle = plt.Circle(center, r, fill=False)
plt.gcf().gca().add_artist(circle)

plt.show()