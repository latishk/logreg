from numpy import genfromtxt
import matplotlib.pyplot as plt
import math

w1 = 1.2
w0 = 0
w2 = -1

def hFunction(x,y):
    return  sigmoid(w1*x+w2*y+w0)

def sigmoid(x):
    return  1/(1+math.exp(-x))

def main():

    alpha = 0.01
    data  = genfromtxt('data1.csv', delimiter=',')
    Xs    = data[:,0]
    Ys    = data[:,1]
    O     = data[:,2]

    sumOfDiffPrevious   = 0
    sumOfSquareDiff     = 0
    counter             = 0
    breakthis           = False
    numberOfEpoches     = 0
    ssd                 =[]
    xaxis               =[]
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

                if i == Ys.size - 1:
                    # square difference checked at the last pass of i.e before the end of the epoch
                    sumOfSquareDiff+= e*e

            global w0, w1, w2
            w0 += alpha*sumOfDiff
            w1 += alpha*sumOfDiffW1
            w2 += alpha*sumOfDiffW2

        numberOfEpoches+=1

        if math.fabs(sumOfSquareDiff - sumOfDiffPrevious) < 0.001:
            counter+=1

        if counter >=2:
            breakthis = True
            numberOfEpoches-=1
            break

        xaxis.append(numberOfEpoches)
        ssd.append(sumOfSquareDiff)
        sumOfDiffPrevious = sumOfSquareDiff
        sumOfSquareDiff = 0


    print(w0,w1,w2,"\n",xaxis,"\n",ssd)
    # plt.plot(ssd,numberOfEpoches)
    print("breakthis", breakthis,"number of epoches",numberOfEpoches,"\n")
    plt.plot(xaxis,ssd)
    plt.scatter(xaxis,ssd)
    plt.show()
    # print(ssd)



main()

print ("w2: ",w2,"w1:",w1,"w0:",w0)





