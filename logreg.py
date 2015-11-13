# from numpy import genfromtxt
import numpy
import matplotlib.pyplot as plt
import math
import csv
import random

random1 = random.uniform(-2, 2)
w1 = random1
random.uniform(-10, 10)
w0 = random1
random.uniform(-10, 10)
w2 = random1

def hFunction(x,y):
    return  sigmoid(w1*x+w2*y+w0)

def sigmoid(x):
    return  1/(1+math.exp(-x))


def main():

    alpha = 0.01
    data  = numpy.genfromtxt('data1.csv', delimiter=',')
    Xs    = data[:,0]
    Ys    = data[:,1]
    O     = data[:,2]

    sumOfDiffPrevious   = 0
    sumOfSquareDiff     = 0
    sumOfDiffSecondLast = 0
    counter             = 0
    breakthis           = False
    numberOfEpoches     = 0
    ssd                 = []
    xaxis               = []
    while numberOfEpoches < 1000:
        for i in range(0, Ys.size):
            sumOfDiff = 0
            sumOfDiffW1 = 0
            sumOfDiffW2 = 0
            for j in range(0,i):

                e = O[j]- hFunction( Xs[j], Ys[j])
                sumOfDiff   += e
                sumOfDiffW1 += e*Xs[j]
                sumOfDiffW2 += e*Ys[j]

                if i == Ys.size - 2 :
                    # square difference checked at the last pass of i.e before the end of the epoch
                    sumOfDiffSecondLast =e*e
                if i == Ys.size - 1:
                    sumOfSquareDiff = e*e

            global w0, w1, w2
            w0 += alpha*sumOfDiff
            w1 += alpha*sumOfDiffW1
            w2 += alpha*sumOfDiffW2

        numberOfEpoches+=1
        # here not calculating the absolute using math.fabs() value since we have to find local minimum and and if the function starts increasing then it shall
        # stop.
        if math.fabs(sumOfDiffSecondLast - sumOfSquareDiff) < 0.001:
            counter+=1

        if counter >=2:
            breakthis = True
            numberOfEpoches-=1
            break

        xaxis.append(numberOfEpoches)
        ssd.append(sumOfSquareDiff)
        sumOfDiffPrevious = sumOfSquareDiff #useless statement

        sumOfSquareDiff = 0

    # print(w0,w1,w2,"\n",xaxis,"\n",ssd)
    # plt.plot(ssd,numberOfEpoches)
    print("number of epoches",numberOfEpoches,"\n")
    plt.plot(xaxis,ssd)
    plt.scatter(xaxis,ssd)
    plt.show()
    testTheFunction('samples.csv')

def yValues(x):
    return -1*( w1/w2 * x + w0/w1 )

def testTheFunction(file):
    sampleData  = numpy.genfromtxt(file, delimiter=',')
    Xs    = sampleData[:,0]
    Ys    = sampleData[:,1]
    classified = numpy.zeros((Ys.size,3))
    correctPrediction = 0

    for i in range(0,Ys.size):
        classified[i,0] = Xs[i]
        classified[i,1] = Ys[i]

        if hFunction(Xs[i],Ys[i]) >= 0.5:
            classified[i,2] = 1
            plt.scatter( Xs[i],Ys[i], marker='1')
        else:
            classified[i,2] = 0
            plt.scatter( Xs[i],Ys[i], marker='o')

        if classified[i,2]==sampleData[i,2]:
            correctPrediction+=1

    print("number of correct predictions: ",correctPrediction,"out of ", Ys.size," which is ", 100*(correctPrediction/Ys.size),"% correct prediction rate")
    maxX = max(Xs)+1
    minX = min(Xs)-1
    # print(maxX,"here",minX)
    t = numpy.arange(minX*1.0, maxX*1.0, 0.5)
    y =[]
    for x in t:
        y.append(yValues(x))
    # print(classified[:,2])
    plt.plot(t,y)
    plt.show()



main()

# b = open('weights.csv', 'w')
# a = csv.writer(b)
#
# a.writerows(weights)
# b.close()
weights = [w0,w1,w2]
file = open('weights.csv','w');
line = '';
for weight in weights:
    line += str(weight) + ",";
file.write(line);
file.close();
print ("w2: ",w2,"w1:",w1,"w0:",w0)



