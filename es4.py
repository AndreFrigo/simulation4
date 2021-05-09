from enum import Enum
import time 
import numpy

class EventQueue:
    def __init__(self):
        self.queue = []

    def insert(self, ev):
        numElem = len(self.queue)
        #control if the queue is empty
        if(numElem == 0):
            self.queue.append(ev)
        else:
            #insert the element in the right position
            i = 0
            index = -1
            while(i<numElem):
                if(self.queue[i].time>ev.time):
                    index = i
                    i = numElem-1
                i += 1
            # if(index>-1):
            #     self.queue.insert(index, ev)
            # else:
            #     self.queue.insert(i, ev)
            if index == -1:
                index = i
            #insert the element
            self.queue.insert(index, ev)
            #link the new element, it is assumed that there exist a start and end event, so each event will have at least one other event before and after itself
            self.queue[index-1].next = self.queue[index]
            self.queue[index].next = self.queue[index+1]

    def toString(self):
        string = ""
        for elem in self.queue:
            string += elem.toString() + ", "
        return string[:-2]


#lambda
lam = 2
#mu
u = 3
#condition to stop the simulation
END = False
#event queue
q = EventQueue()
#time
t = 0
#maximum simulation time (s)
maxTime = 1000
#seed of the RNG
numpy.random.seed(23)
#server status: true if it is free, false if it is working
server = True
#number of packets in the queue
packetQueue = 0

class Type(Enum):
    START = 1
    END = 2
    ARRIVAL = 3
    DEPARTURE = 4
    DEBUG = 5

class Event:
    def __init__(self, time, t):
        self.time = time
        self.type = t
        self.next = None
    def toString(self):
        return(str(self.time)+"-"+str(self.type.name))
    def execute(self, queue):
        typ = self.type.value
        global t
        if typ == 1:
            print("START")
            #schedule an arrival event
            #global t
            t += numpy.random.exponential(1/lam)
            global q
            q.insert(Event(t, Type.ARRIVAL))
            # e.execute(queue)
        elif typ == 2:
            print("END")
            global END
            END = True
        elif typ == 3:
            print("ARRIVAL")
            #schedule the next arrival
            at = self.time+numpy.random.exponential(1/lam)
            if(at<maxTime):
                q.insert(Event(at, Type.ARRIVAL))
            #if server is free, seize it and schedule the departure of the packet
            global server
            if server == True:
                server = False
                at = self.time+numpy.random.exponential(1/u)
                if(at<maxTime):
                    q.insert(Event(at, Type.DEPARTURE))
                #self.execute(queue)
            #if the server is busy increase the number of packet in the queue
            else:
                packetQueue += 1
        elif typ == 4:
            print("DEPARTURE")
            if(packetQueue == 0):
                global server
                server = True 
            else:
                server = False
                at = self.time+numpy.random.exponential(1/u)
                if (at<maxTime):
                    q.insert(Event(at, Type.DEPARTURE))
        elif typ == 5:
            print(f"DEBUG TIME: {t}")
            print(queue.toString())









# q = EventQueue()
# e = Event(0,Type.ARRIVAL)
# q.insert(e)
# e = Event(7,Type.DEBUG)
# q.insert(e)
# e = Event(12,Type.DEPARTURE)
# q.insert(e)
# e = Event(18,Type.END)
# q.insert(e)
# e = Event(21,Type.ARRIVAL)
# q.insert(e)
# print(f"Before insert 16: {q.toString()}")
# e = Event(16,Type.DEPARTURE)
# q.insert(e)
# print(f"After insert 16: {q.toString()}")
# ### Insert ok 



#insert start and end event
start = Event(0, Type.START)
end = Event(1000, Type.END)
start.next = END
q.queue.append(start)
q.queue.append(end)
print(q.toString())

# # insert some events out of order
# q.insert(Event(150, Type.ARRIVAL))
# q.insert(Event(76, Type.DEPARTURE))
# q.insert(Event(120, Type.DEBUG))
# q.insert(Event(104, Type.DEPARTURE))
# q.insert(Event(100, Type.ARRIVAL))
# q.insert(Event(20, Type.DEPARTURE))
# q.insert(Event(250, Type.DEBUG))
# q.insert(Event(66, Type.DEPARTURE))

#event manager loop
while(not END):
    #execute event (first the start event)
    ev = q.queue.pop(0)
    ev.execute(q)
    
