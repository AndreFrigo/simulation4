from enum import Enum
import time 
import numpy
import matplotlib.pyplot as plt


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
            # self.queue[index-1].next = self.queue[index]
            # self.queue[index].next = self.queue[index+1]

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
#maximum simulation time (s)  ##500000
maxTime = 10
#seed of the RNG
numpy.random.seed(232)
#server status: true if it is free, false if it is working
server = True
#number of packets in the queue
packetQueue = 0

#part2
e2c = 2
e2k = 10
servers = []
for i in range(0,e2c):
    servers.append([True, []])
    
#es 2, packet assignment policy used
assignmentPolicy = "round-robin"
contEv = 0

def rr(n, servers):
    return n % servers

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
        global packetQueue
        if typ == 1:
            print("START")
            #schedule an arrival event
            #global t
            at = numpy.random.exponential(1/lam)
            # global q
            q.insert(Event(at, Type.ARRIVAL))
            # e.execute(queue)
        elif typ == 2:
            print("END")
            global END
            END = True
        elif typ == 3:
            # print(f"ARRIVAL t: {self.time}")
            #schedule the next arrival
            at = self.time+numpy.random.exponential(1/lam)
            q.insert(Event(at, Type.ARRIVAL))
            #if server is free, seize it and schedule the departure of the packet
            global server
            if server == True:
                server = False
                at = self.time+numpy.random.exponential(1/u)
                q.insert(Event(at, Type.DEPARTURE))
                #self.execute(queue)
            #if the server is busy increase the number of packet in the queue
            else:
                # global packetQueue
                packetQueue += 1
        elif typ == 4:
            # print(f"DEPARTURE t: {self.time}")
            # global server
            if(packetQueue == 0):
                server = True 
            else:
                server = False
                at = self.time+numpy.random.exponential(1/u)
                q.insert(Event(at, Type.DEPARTURE))
                # global packetQueue
                packetQueue -= 1
        elif typ == 5:
            print(f"   DEBUG TIME: {self.time}")
            # print(queue.toString())
            print(f"      PacketQueue: {packetQueue}, server: {server}")


#for es2
class Event2:
    def __init__(self, time, t):
        self.time = time
        self.type = t
        self.next = None
    def toString(self):
        return(str(self.time)+"-"+str(self.type.name))
    def execute(self, queue):
        typ = self.type.value
        global packetQueue
        global contEv
        if typ == 1:
            print("START")
            at = numpy.random.exponential(1/lam)
            q.insert(Event2(at, Type.ARRIVAL))
        elif typ == 2:
            print("END")
            global END
            END = True
        elif typ == 3:
            print("ARRIVAL")
            at = self.time+numpy.random.exponential(1/lam)
            q.insert(Event2(at, Type.ARRIVAL))
            #if server is free, seize it and schedule the departure of the packet
            if(assignmentPolicy == "round-robin"):
                s = rr(contEv, e2c)
                contEv += 1
                print(f"Server {servers[s][0]}")
                if servers[s][0] == True:
                    print(f"modifica server {s}")
                    servers[s][0] = False
                    at = self.time+numpy.random.exponential(1/u)
                    q.insert(Event2(at, Type.DEPARTURE))
                #if the server is busy increase the number of packet in the queue
                else:
                    #if queue is not full
                    if(len(servers[s][1]) < e2k):
                        servers[s][1].append(self.time)
            print(servers)
        elif typ == 4:
            print("DEPARTURE")
            if(assignmentPolicy == "round-robin"):
                s = rr(contEv, e2c)
                contEv += 1
                if(len(servers[s][1]) == 0):
                    servers[s][0] = True
                else:
                    servers[s][0] = False
                    at = self.time+numpy.random.exponential(1/u)
                    q.insert(Event2(at, Type.DEPARTURE))
                    servers[s][1].pop(0)
            print(server)
        elif typ == 5:
            print(f"   DEBUG TIME: {self.time}")
            # print(queue.toString())
            print(f"      PacketQueue: {packetQueue}, server: {server}")



#mean given dataset data
def mean(data):
    m = 0
    n = len(data)
    for elem in data:
        m+=float(elem)
    return m/n

#insert start and end event
start = Event(0, Type.START)
end = Event(maxTime, Type.END)
start.next = END
q.queue.append(start)
q.queue.append(end)
# print(q.toString())

# # insert some events out of order
# q.insert(Event(150, Type.ARRIVAL))
# q.insert(Event(76, Type.DEPARTURE))
# q.insert(Event(120, Type.DEBUG))
# q.insert(Event(104, Type.DEPARTURE))
# q.insert(Event(100, Type.ARRIVAL))
# q.insert(Event(20, Type.DEPARTURE))
# q.insert(Event(250, Type.DEBUG))
# q.insert(Event(66, Type.DEPARTURE))


time = []
packetNumber = []
queueBusy = []
t = 0
np = 0

#event manager loop
# waitingTime = EventQueue()
waitingTime = []
wt = []
# while(not END):
#     #execute event (first the start event)
#     time.append(t)
#     packetNumber.append(np)
#     ev = q.queue.pop(0)
#     # print(f"EVENT: {ev.type}, TIME: {ev.time}, NP: {np}, SERVER: {server}, PACKETQUEUE: {packetQueue}")
#     # #se tipo arrival e server occupato salva time, al momento della prima departure calcola delta t
#     if(ev.type == Type.ARRIVAL and server):
#         #wt.append(0)
#         waitingTime.append(-1)
#         # print("APPEND -1")
#     if(ev.type == Type.ARRIVAL and (not server)):
#         #waitingTime.insert(Event(ev.time, Type.ARRIVAL))
#         waitingTime.append(ev.time)
#         # print(f"APPEND {ev.time}")
#     if(ev.type == Type.DEPARTURE):
#         #rem = waitingTime.queue.pop(0)
#         if(len(waitingTime) > 0):
#             rem = waitingTime.pop(0)
#             # print(f"POP {rem}")
#             # wt.append(rem.time - ev.time)
#             if(rem != -1):
#                 wt.append(ev.time - rem)
#             elif(len(waitingTime) > 0):
#                 wt.append(0)
#                 r2 = waitingTime.pop(0)
#                 wt.append(ev.time - r2)
#             else:
#                 wt.append(0)

#     # print(waitingTime)
#     # print(wt)
#     ev.execute(q)
    
#     # if(ev.type != Type.DEBUG):
#     #     q.insert(Event(ev.time + 0.01, Type.DEBUG))

#     t = ev.time
#     if (server):
#         np = packetQueue
#     else:
#         np = packetQueue + 1

# # print(time)
# # print(packetNumber)


# i = 0
# j = 0
# packetNumberInt = []
# timeInt = []
# interval = 10000
# print(f"Len packetNumber: {len(packetNumber)}, len time: {len(time)}, len wt: {len(wt)}")

    
# y = (lam/u)/(1-(lam/u))
# # print(y)


# # # # i = 1
# # # # mean = 0
# # # # while i<len(time):
# # # #     mean += (time[i]-time[i-1])*packetNumber[i-1]
# # # #     i += 1
# # # # print(mean/maxTime)

# # interval = 500
# # intervals = list(range(interval, len(time), interval))
# # intervals.append(len(time))
# # means = []
# # for elem in intervals:
# #     i = 1
# #     mean = 0
# #     while i<elem:
# #         mean += (time[i]-time[i-1])*packetNumber[i-1]
# #         i += 1
# #     means.append(mean/time[elem-1])

# # times = []
# # for elem in intervals:
# #     times.append(time[elem -1])
# # plt.plot(times, means)
# # plt.show()




# #print(mean(wt))
# y = ((lam/u)**2) / (lam*(1-(lam/u)))
# #print(y)
# print(f"Average time that a packet has to wait in the queue: {mean(wt)}, expected: {y}")







#part 2

q = EventQueue()
#insert start and end event
start = Event2(0, Type.START)
end = Event2(maxTime, Type.END)
q.queue.append(start)
q.queue.append(end)
# print(q.toString())

packetServed = []
waitingTime = []
wt = []
for i in range(0, e2c):
    waitingTime.append([])
    wt.append([])
    packetServed.append(0)

while(not END):
    #execute event (first the start event)
    time.append(t)
    packetNumber.append(np)
    ev = q.queue.pop(0)
    # print(f"EVENT: {ev.type}, TIME: {ev.time}, NP: {np}, SERVER: {server}, PACKETQUEUE: {packetQueue}")
    # #se tipo arrival e server occupato salva time, al momento della prima departure calcola delta t
    s = rr(contEv, e2c)
    if(ev.type == Type.ARRIVAL and server):
        #wt.append(0)
        waitingTime[s].append(-1)
        # print("APPEND -1")
    if(ev.type == Type.ARRIVAL and (not server)):
        #waitingTime.insert(Event(ev.time, Type.ARRIVAL))
        waitingTime[s].append(ev.time)
        # print(f"APPEND {ev.time}")
    if(ev.type == Type.DEPARTURE):
        #rem = waitingTime.queue.pop(0)
        if(len(waitingTime[s]) > 0):
            rem = waitingTime[s].pop(0)
            # print(f"POP {rem}")
            # wt.append(rem.time - ev.time)
            if(rem != -1):
                wt[s].append(ev.time - rem)
                packetServed[s] += 1
            elif(len(waitingTime[s]) > 0):
                wt[s].append(0)
                r2 = waitingTime[s].pop(0)
                wt[s].append(ev.time - r2)
                packetServed[s] += 2
            else:
                wt[s].append(0)
                packetServed[s] += 1

    ev.execute(q)

print(servers)
#waiting time for each server
print(wt)
#number of packet served from each server
print(packetServed)
