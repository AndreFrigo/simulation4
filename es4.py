from enum import Enum
import time 
import numpy
import matplotlib.pyplot as plt
import math
import random


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
u = 5
#condition to stop the simulation
END = False
#event queue
q = EventQueue()
#time
t = 0
#maximum simulation time (s)
maxTime = 300000
#seed of the RNG
numpy.random.seed(232)
#server status: true if it is free, false if it is working
server = True
#number of packets in the queue
packetQueue = 0

#part2
e2c = 10
e2k = 10
servers = []
for i in range(0,e2c):
    servers.append([True, []])
    
#es 2, packet assignment policy used
class AssignmentPolicy(Enum):
    ROUND_ROBIN = 1
    LEAST_LOADED = 2
    QUEUE_OCCUPANCY = 3

contEv = 0

#mean given dataset data
def mean(data):
    m = 0
    n = len(data)
    for elem in data:
        m+=float(elem)
    return m/n

#variance given dataset data and mean
def variance(data, m=0):
    var = 0
    n = len(data)
    if m == 0:
        m = mean(data)
    for elem in data:
        var += (float(elem) - m)**2
    return var/n

#round robin
def rr(n, servers):
    return n % servers

#least loaded
def leastLoaded(servers):
    minSer = [-1, e2k+1]
    for i in range(0, e2c):
        if servers[i][0] == True:
            return i
        elif len(servers[i][1]) < minSer[1]:
            minSer[0] = i
            minSer[1] = len(servers[i][1])
    # print(f"MIN SER: {minSer[0]}")
    return minSer[0]

def mostLoaded(servers):
    maxSer = [-1, -1]
    for i in range(0, e2c):
        if servers[i][0] == False and len(servers[i][1]) >= maxSer[1]:
            maxSer[0] = i
            maxSer[1] = len(servers[i][1])
    # print(f"MAX SER: {maxSer[0]}")
    return maxSer[0]

#avoid sending packets to a queue that is more than x% full
def queueOccupancy(x, servers):
    maxQ = math.floor(x*e2k/100)
    for i in range(0, e2c):
        if(len(servers[i][1]) < maxQ):
            return i
    #if no queue satisfies the condition, choose one randomly 
    return random.randint(0,e2c-1)

def mean19(dataBootstrap, data, time):
    e = time[len(time)-1]
    b = time[0]
    i = 1
    mean19 = 0
    while i<len(data):
        mean19 += (time[dataBootstrap[i]]-time[dataBootstrap[i]-1])*data[dataBootstrap[i]-1]
        i += 1
    return mean19/(e-b)

#mean given dataset data
def mean111(dataBootstrap, data, time):
    m = 0
    n = len(data)
    for elem in data:
        m+=float(elem)
    return m/n

#bootstrap given dataset data and time, confidence interval level cil, algorithm's accuracy parameter r0, the statistical function to apply fun 
def bootstrap(data, time, cil, r0, fun):
    dataBootstrap = []
    res = []
    n = len(data)
    rep = math.ceil(2*25/(1-cil))-1
    i = 0
    for z in range(rep):
        for x in range(n):
            r = random.randint(1, n-1)
            dataBootstrap.append(r) 
        
        #calculate function and insert in the list res
        res.append(fun(dataBootstrap, data, time))    
        dataBootstrap = []
        i += 1
        print(f"Bootstrap cicle {i} of {rep}")

    res.sort()
    #return prediction interval
    return [res[r0], res[rep+1-r0]]   

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
        if(assignmentPolicy == AssignmentPolicy.ROUND_ROBIN):
            s = rr(contEv, e2c)
            contEv += 1
        elif(assignmentPolicy == AssignmentPolicy.LEAST_LOADED):
            s = leastLoaded(servers)
        elif(assignmentPolicy == AssignmentPolicy.QUEUE_OCCUPANCY):
            s = queueOccupancy(qOcc, servers)
        if typ == 1:
            print("START")
            at = numpy.random.exponential(1/lam)
            q.insert(Event2(at, Type.ARRIVAL))
        elif typ == 2:
            print("END")
            global END
            END = True
        elif typ == 3:
            # print("ARRIVAL")
            at = self.time+numpy.random.exponential(1/lam)
            q.insert(Event2(at, Type.ARRIVAL))
            #if server is free, seize it and schedule the departure of the packet
            
            #print(f"Server {servers[s][0]}")
            if servers[s][0] == True:
                servers[s][0] = False
                at = self.time+numpy.random.exponential(1/u)
                q.insert(Event2(at, Type.DEPARTURE))
            #if the server is busy increase the number of packet in the queue
            else:
                #if queue is not full
                if(len(servers[s][1]) < e2k):
                    servers[s][1].append(self.time)
            #print(servers)
        elif typ == 4:
            # print("DEPARTURE")
            
            if(len(servers[s][1]) == 0):
                servers[s][0] = True
            else:
                servers[s][0] = False
                at = self.time+numpy.random.exponential(1/u)
                q.insert(Event2(at, Type.DEPARTURE))
                servers[s][1].pop(0)
            #print(server)
        elif typ == 5:
            print(f"   DEBUG TIME: {self.time}")
            # print(queue.toString())
            print(f"      PacketQueue: {packetQueue}, server: {server}")
        # print(f"Servers: {servers}")
        # print(f"S: {s}")





# #insert start and end event
# start = Event(0, Type.START)
# end = Event(maxTime, Type.END)
# start.next = END
# q.queue.append(start)
# q.queue.append(end)
# # print(q.toString())

# # # # insert some events out of order
# # # q.insert(Event(150, Type.ARRIVAL))
# # # q.insert(Event(76, Type.DEPARTURE))
# # # q.insert(Event(120, Type.DEBUG))
# # # q.insert(Event(104, Type.DEPARTURE))
# # # q.insert(Event(100, Type.ARRIVAL))
# # # q.insert(Event(20, Type.DEPARTURE))
# # # q.insert(Event(250, Type.DEBUG))
# # # q.insert(Event(66, Type.DEPARTURE))


# time = []
# packetNumber = []
# queueBusy = []
# t = 0
# np = 0

# #event manager loop
# # waitingTime = EventQueue()
# waitingTime = []
# wt = []
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
# END = False

# i = 0
# j = 0
# packetNumberInt = []
# timeInt = []
# # print(f"Len packetNumber: {len(packetNumber)}, len time: {len(time)}, len wt: {len(wt)}")

# packetNumber = packetNumber[200000:len(packetNumber)]
# wt = wt[200000:len(wt)]
# time = time[200000:len(time)]

# ###### 1.9
    
# y = (lam/u)/(1-(lam/u))
# oldMaxTime = maxTime
# maxTime = 100000
# i = 1
# mean1 = 0
# while i<len(time):
#     mean1 += (time[i]-time[i-1])*packetNumber[i-1]
#     i += 1
# ci = 0
# ci = bootstrap(packetNumber, time, 0.95, 25, mean19)
# print(f"Average number of packets in the system: {mean1/(time[len(time)-1]-time[0])}, with 95%ci : {ci}, expected: {y}")

# interval = 2000
# intervals = list(range(interval, len(time), interval))
# intervals.append(len(time))
# means = []
# for elem in intervals:
#     i = 1
#     mean2 = 0
#     while i<elem:
#         mean2 += (time[i]-time[i-1])*packetNumber[i-1]
#         i += 1
#     means.append(mean2/(time[elem-1]-time[0]))

# times = []
# for elem in intervals:
#     times.append(time[elem -1])
# plt.plot(times, means)
# plt.plot([0, maxTime], [y, y], color='red')
# title = "Number of packets in the system over time, with lam: " + str(lam) + ", mu: " + str(u)
# plt.title(title)
# plt.show()
# maxTime = oldMaxTime

# ####finish 1.9

# ###### 1.11
# y = ((lam/u)**2) / (lam*(1-(lam/u)))
# ci = 0
# mean1 = mean(wt)
# var = variance(wt, mean1)
# numElem = len(wt)
# ci = [mean1 - (1.96 * math.sqrt(var) / math.sqrt(numElem)), mean1 + (1.96 * math.sqrt(var) / math.sqrt(numElem))]

# print(f"Average time that a packet has to wait in the queue: {mean(wt)}, with 95% ci: {ci}, expected: {y}")



###### finish 1.11

# ####### 1.12
# weights = numpy.ones_like(packetNumber)/float(len(packetNumber))
# plt.hist(packetNumber, weights=weights, bins=14)
# # plt.hist(packetNumber, bins=14, density=True)
# plt.title("Empirical distribution of the number of packets in the system")
# plt.show()


# weights = numpy.ones_like(wt)/float(len(wt))
# plt.hist(wt, weights=weights, bins=14)
# #plt.hist(wt, bins=14, density=True)
# plt.title("Empirical distribution of the queue waiting time")
# plt.show()
# #### finish 1.12


# #part 2


###2.1


# q = EventQueue()
# #insert start and end event
# start = Event2(0, Type.START)
# end = Event2(maxTime, Type.END)
# q.queue.append(start)
# q.queue.append(end)
# # print(q.toString())

# packetServed = []
# waitingTime = []
# wt = []
# for i in range(0, e2c):
#     waitingTime.append([])
#     wt.append([])
#     packetServed.append(0)
# time = []
# packetNumber = []
# queueBusy = []
# t = 0
# np = 0


# #round robin
# print("\n\nRound robin")
# assignmentPolicy = AssignmentPolicy.ROUND_ROBIN
# while(not END):
#     #execute event (first the start event)
#     time.append(t)
#     packetNumber.append(np)
#     ev = q.queue.pop(0)
#     # print(f"EVENT: {ev.type}, TIME: {ev.time}, NP: {np}, SERVER: {server}, PACKETQUEUE: {packetQueue}")
#     # #se tipo arrival e server occupato salva time, al momento della prima departure calcola delta t
#     s = rr(contEv, e2c)
#     if(ev.type == Type.ARRIVAL and server):
#         #wt.append(0)
#         waitingTime[s].append(-1)
#         # print("APPEND -1")
#     if(ev.type == Type.ARRIVAL and (not server)):
#         #waitingTime.insert(Event(ev.time, Type.ARRIVAL))
#         waitingTime[s].append(ev.time)
#         # print(f"APPEND {ev.time}")
#     if(ev.type == Type.DEPARTURE):
#         #rem = waitingTime.queue.pop(0)
#         if(len(waitingTime[s]) > 0):
#             rem = waitingTime[s].pop(0)
#             # print(f"POP {rem}")
#             # wt.append(rem.time - ev.time)
#             if(rem != -1):
#                 wt[s].append(ev.time - rem)
#                 packetServed[s] += 1
#             elif(len(waitingTime[s]) > 0):
#                 wt[s].append(0)
#                 r2 = waitingTime[s].pop(0)
#                 wt[s].append(ev.time - r2)
#                 packetServed[s] += 2
#             else:
#                 wt[s].append(0)
#                 packetServed[s] += 1
#     np = 0
#     for elem in servers:
#         if elem[0] == False:
#             np += 1
#         np += len(elem[1])
#     ev.execute(q)
#     t = ev.time

# # print(servers)
# # #waiting time for each server
# # print(wt)
# #number of packet served from each server
# # print(packetServed)
# END = False

# ### 1.9
    
# y = (lam/u)/(1-(lam/u))
# oldMaxTime = maxTime
# maxTime = 100000
# i = 1
# mean1 = 0
# while i<len(time):
#     mean1 += (time[i]-time[i-1])*packetNumber[i-1]
#     i += 1
# ci = 0
# ci = bootstrap(packetNumber, time, 0.95, 25, mean19)
# print(f"Average number of packets in the system: {mean1/(time[len(time)-1]-time[0])}, with 95%ci : {ci}")

# interval = 2000
# intervals = list(range(interval, len(time), interval))
# intervals.append(len(time))
# means = []
# for elem in intervals:
#     i = 1
#     mean2 = 0
#     while i<elem:
#         mean2 += (time[i]-time[i-1])*packetNumber[i-1]
#         i += 1
#     means.append(mean2/(time[elem-1]-time[0]))

# times = []
# for elem in intervals:
#     times.append(time[elem -1])
# plt.plot(times, means)
# title = "Number of packets in the system over time, with lam: " + str(lam) + ", mu: " + str(u) +", c: " + str(e2c) + ", K: " + str(e2k)
# plt.title(title)
# plt.show()
# maxTime = oldMaxTime


# ## 1.11
# y = ((lam/u)**2) / (lam*(1-(lam/u)))
# ci = 0
# wt11 = []
# for elem in wt:
#     for e in elem:
#         wt11.append(e)

# mean1 = mean(wt11)
# var = variance(wt11, mean1)
# numElem = len(wt11)
# ci = [mean1 - (1.96 * math.sqrt(var) / math.sqrt(numElem)), mean1 + (1.96 * math.sqrt(var) / math.sqrt(numElem))]

# print(f"Average time that a packet has to wait in the queue: {mean1}, with 95% ci: {ci}")




###2.2

# q = EventQueue()
# #insert start and end event
# start = Event2(0, Type.START)
# end = Event2(maxTime, Type.END)
# q.queue.append(start)
# q.queue.append(end)
# # print(q.toString())

# packetServed = []
# waitingTime = []
# wt = []
# for i in range(0, e2c):
#     waitingTime.append([])
#     wt.append([])
#     packetServed.append(0)
# time = []
# packetNumber = []
# queueBusy = []
# t = 0
# np = 0


# #round robin
# print("\n\nRound robin")
# assignmentPolicy = AssignmentPolicy.ROUND_ROBIN
# while(not END):
#     #execute event (first the start event)
#     time.append(t)
#     packetNumber.append(np)
#     ev = q.queue.pop(0)
#     # print(f"EVENT: {ev.type}, TIME: {ev.time}, NP: {np}, SERVER: {server}, PACKETQUEUE: {packetQueue}")
#     # #se tipo arrival e server occupato salva time, al momento della prima departure calcola delta t
#     s = rr(contEv, e2c)
#     if(ev.type == Type.ARRIVAL and server):
#         #wt.append(0)
#         waitingTime[s].append(-1)
#         # print("APPEND -1")
#     if(ev.type == Type.ARRIVAL and (not server)):
#         #waitingTime.insert(Event(ev.time, Type.ARRIVAL))
#         waitingTime[s].append(ev.time)
#         # print(f"APPEND {ev.time}")
#     if(ev.type == Type.DEPARTURE):
#         #rem = waitingTime.queue.pop(0)
#         if(len(waitingTime[s]) > 0):
#             rem = waitingTime[s].pop(0)
#             # print(f"POP {rem}")
#             # wt.append(rem.time - ev.time)
#             if(rem != -1):
#                 wt[s].append(ev.time - rem)
#                 packetServed[s] += 1
#             elif(len(waitingTime[s]) > 0):
#                 wt[s].append(0)
#                 r2 = waitingTime[s].pop(0)
#                 wt[s].append(ev.time - r2)
#                 packetServed[s] += 2
#             else:
#                 wt[s].append(0)
#                 packetServed[s] += 1

#     ev.execute(q)

# # print(servers)
# # #waiting time for each server
# # print(wt)
# #number of packet served from each server
# # print(packetServed)
# END = False

# fig, axes = plt.subplots(figsize=(7,5), dpi=100)
# plt.bar(range(1, e2c+1), height=packetServed)
# title = 'Packet served by each server using RR, c:' + str(e2c) + ', k: ' + str(e2k)
# plt.title(title)
# plt.show()

# ywt = []
# for elem in wt:
#     ywt.append(mean(elem))
# fig, axes = plt.subplots(figsize=(7,5), dpi=100)
# plt.bar(range(1, e2c+1), height=ywt)
# title = 'Queuing delay using RR, c:' + str(e2c) + ', k: ' + str(e2k)
# plt.title(title)
# plt.show()







# servers = []
# for i in range(0,e2c):
#     servers.append([True, []])
# q = EventQueue()
# #insert start and end event
# start = Event2(0, Type.START)
# end = Event2(maxTime, Type.END)
# q.queue.append(start)
# q.queue.append(end)
# # print(q.toString())
# time = []
# packetNumber = []
# queueBusy = []
# t = 0
# np = 0
# packetServed = []
# waitingTime = []
# wt = []
# for i in range(0, e2c):
#     waitingTime.append([])
#     wt.append([])
#     packetServed.append(0)
# #least loaded
# print("\n\nLeast loaded")
# assignmentPolicy = AssignmentPolicy.LEAST_LOADED
# while(not END):
#     #execute event (first the start event)
#     time.append(t)
#     packetNumber.append(np)
#     ev = q.queue.pop(0)
#     # print(f"EVENT: {ev.type}, TIME: {ev.time}, NP: {np}, SERVER: {server}, PACKETQUEUE: {packetQueue}")
#     # #se tipo arrival e server occupato salva time, al momento della prima departure calcola delta t
#     s = leastLoaded(servers)
#     s2 = mostLoaded(servers)
#     if(ev.type == Type.ARRIVAL and server):
#         #wt.append(0)
#         waitingTime[s].append(-1)
#         # print("APPEND -1")
#     if(ev.type == Type.ARRIVAL and (not server)):
#         #waitingTime.insert(Event(ev.time, Type.ARRIVAL))
#         waitingTime[s].append(ev.time)
#         # print(f"APPEND {ev.time}")
#     if(ev.type == Type.DEPARTURE):
#         #rem = waitingTime.queue.pop(0)
#         if(len(waitingTime[s2]) > 0):
#             rem = waitingTime[s2].pop(0)
#             # print(f"POP {rem}")
#             # wt.append(rem.time - ev.time)
#             if(rem != -1):
#                 wt[s2].append(ev.time - rem)
#                 packetServed[s2] += 1
#             elif(len(waitingTime[s2]) > 0):
#                 wt[s2].append(0)
#                 r2 = waitingTime[s2].pop(0)
#                 wt[s2].append(ev.time - r2)
#                 packetServed[s2] += 2
#             else:
#                 wt[s2].append(0)
#                 packetServed[s2] += 1

#     ev.execute(q)

# # print(servers)
# # #waiting time for each server
# # print(wt)
# #number of packet served from each server
# # print(packetServed)
# END = False
# fig, axes = plt.subplots(figsize=(7,5), dpi=100)
# plt.bar(range(1, e2c+1), height=packetServed)
# title = 'Packet served by each server using Least Loaded, c:' + str(e2c) + ', k: ' + str(e2k)
# plt.title(title)
# plt.show()

# ywt = []
# for elem in wt:
#     ywt.append(mean(elem))
# fig, axes = plt.subplots(figsize=(7,5), dpi=100)
# plt.bar(range(1, e2c+1), height=ywt)
# title = 'Queuing delay using Least Loaded, c:' + str(e2c) + ', k: ' + str(e2k)
# plt.title(title)
# plt.show()





# servers = []
# for i in range(0,e2c):
#     servers.append([True, []])
# q = EventQueue()
# #insert start and end event
# start = Event2(0, Type.START)
# end = Event2(maxTime, Type.END)
# q.queue.append(start)
# q.queue.append(end)
# # print(q.toString())

# packetServed = []
# waitingTime = []
# wt = []
# for i in range(0, e2c):
#     waitingTime.append([])
#     wt.append([])
#     packetServed.append(0)
# time = []
# packetNumber = []
# queueBusy = []
# t = 0
# np = 0
# qOcc = 30
# #avoid sending packets to a queue that is more than 30% full
# assignmentPolicy = AssignmentPolicy.QUEUE_OCCUPANCY
# print("\n\nQueue occupancy")
# while(not END):
#     #execute event (first the start event)
#     time.append(t)
#     packetNumber.append(np)
#     ev = q.queue.pop(0)
#     # print(f"EVENT: {ev.type}, TIME: {ev.time}, NP: {np}, SERVER: {server}, PACKETQUEUE: {packetQueue}")
#     # #se tipo arrival e server occupato salva time, al momento della prima departure calcola delta t
#     s = queueOccupancy(qOcc, servers)
#     if(ev.type == Type.ARRIVAL and server):
#         #wt.append(0)
#         waitingTime[s].append(-1)
#         # print("APPEND -1")
#     if(ev.type == Type.ARRIVAL and (not server)):
#         #waitingTime.insert(Event(ev.time, Type.ARRIVAL))
#         waitingTime[s].append(ev.time)
#         # print(f"APPEND {ev.time}")
#     if(ev.type == Type.DEPARTURE):
#         #rem = waitingTime.queue.pop(0)
#         if(len(waitingTime[s]) > 0):
#             rem = waitingTime[s].pop(0)
#             # print(f"POP {rem}")
#             # wt.append(rem.time - ev.time)
#             if(rem != -1):
#                 wt[s].append(ev.time - rem)
#                 packetServed[s] += 1
#             elif(len(waitingTime[s]) > 0):
#                 wt[s].append(0)
#                 r2 = waitingTime[s].pop(0)
#                 wt[s].append(ev.time - r2)
#                 packetServed[s] += 2
#             else:
#                 wt[s].append(0)
#                 packetServed[s] += 1

#     ev.execute(q)

# # print(servers)
# # #waiting time for each server
# # print(wt)
# #number of packet served from each server
# # print(packetServed)
# END = False
# fig, axes = plt.subplots(figsize=(7,5), dpi=100)
# plt.bar(range(1, e2c+1), height=packetServed)
# title = 'Packet served by each server with control for ' + str(qOcc) +'% of the queue, c:' + str(e2c) + ', k: ' + str(e2k)
# plt.title(title)
# plt.show()

# ywt = []
# for elem in wt:
#     ywt.append(mean(elem))
# fig, axes = plt.subplots(figsize=(7,5), dpi=100)
# plt.bar(range(1, e2c+1), height=ywt)
# title = 'Queuing delay with control for ' + str(qOcc) +'% of the queue, c:' + str(e2c) + ', k: ' + str(e2k)
# plt.title(title)
# plt.show()



# #2.4 
lam = 6.5
#mu
u = 3
#condition to stop the simulation
END = False
#event queue
q = EventQueue()
#time
t = 0
# # #maximum simulation time (s)  
maxTime = 60000
# #seed of the RNG
# numpy.random.seed(232)
#server status: true if it is free, false if it is working
server = True
#number of packets in the queue
packetQueue = 0

e2c = 3
e2k = 2

pNum = 0

q = EventQueue()
#insert start and end event
start = Event2(0, Type.START)
end = Event2(maxTime, Type.END)
q.queue.append(start)
q.queue.append(end)
# print(q.toString())

time = []
packetNumber = []
queueBusy = []
t = 0
np = 0

packetServed = []
waitingTime = []
wt = []
for i in range(0, e2c):
    waitingTime.append([])
    wt.append([])
    packetServed.append(0)
print(f"PACKET SERVED INIT: {waitingTime}")
assignmentPolicy = AssignmentPolicy.ROUND_ROBIN
contEv = 0

while(not END):
    #execute event (first the start event)
    time.append(t)
    packetNumber.append(np)
    ev = q.queue.pop(0)
    # print(f"EVENT: {ev.type}, TIME: {ev.time}, NP: {np}, SERVER: {server}, PACKETQUEUE: {packetQueue}")
    # #se tipo arrival e server occupato salva time, al momento della prima departure calcola delta t
    s = rr(contEv, e2c)
    if(ev.type == Type.ARRIVAL and server):
        pNum += 1
        #wt.append(0)
        waitingTime[s].append(-1)
        # print("APPEND -1")
    if(ev.type == Type.ARRIVAL and (not server)):
        #waitingTime.insert(Event(ev.time, Type.ARRIVAL))
        pNum += 1
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
print(f"SUM packetServed: {sum(packetServed)}")
packetDropProb = 1 - float(sum(packetServed)/pNum)
numElem = len(packetServed)
mean1 = mean(packetServed)
var = variance(packetServed, mean1)
c = [mean1 - (1.96 * math.sqrt(var) / math.sqrt(numElem)), mean1 + (1.96 * math.sqrt(var) / math.sqrt(numElem))]
# print(f"MEAN: {mean1}, var: {var}, ci: {c}")
# print(f"mean nuova {mean1*len(packetServed)/pNum}")
ci = [1-c[1]*numElem/pNum, 1-c[0]*numElem/pNum]
print(f"With lam: {lam}, c: {e2c}, mu: {u}, k: {e2k}, probability of packet drop: {packetDropProb}, ci: {ci}")

res.append(packetDropProb)
