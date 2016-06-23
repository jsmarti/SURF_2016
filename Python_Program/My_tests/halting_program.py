'''This is a halting program that prompts for inputs while executing'''
import numpy as np
import pickle as pkl
import sys
import threading
import time

class Worker(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.current_status = 'Created'
        self.setDaemon(True)
        self.finished = False

    def get_current_status(self):
        return self.current_status

    def get_finished(self):
        return self.finished

    def run(self):
        for i in range(5):
            time.sleep(2)
            self.current_status = 'Iteration ' + str(i) + ' of 5'
        self.finished = True


class Averager:
    def __init__(self):
        pass

    def prompt(self):
        iterations = int(input("Maximum number of runs?: "))
        array = np.zeros(iterations)
        for i in range(iterations):
            data = float(input("Data ?: "))
            array[i] = data
        return np.average(array)

if __name__ == "__main__":
#    option = int(sys.argv[1])
#    if option == 0:
#        print 'A new model will be created'
#        avg = Averager()
#        average = avg.prompt()
#        print 'Average of data: ', average
#        f = open('averager.obj','w')
#        pkl.dump(avg, f, pkl.HIGHEST_PROTOCOL)
#        print 'Averager serialized'
#    elif option == 1:
#        print 'Loading previous model'
#        f = open('averager.obj','r')
#        avg = pkl.load(f)
#        average = avg.prompt()
#        print 'Average of data: ', average

    worker = Worker()
    worker.start()

    while not worker.get_finished():
        print worker.get_current_status()
