__author__ = 'wenbo'
import spynnaker.pyNN as sim
import pylab
import scipy.io
import bluetooth
import numpy as np
import matplotlib.pyplot as plt
import bluetooth
import sys


cell_params_lif = {
                   'cm': 0.3,
                   'i_offset': 0.0,
                   'tau_m': 7,
                   'tau_refrac': 20.0,
                   'tau_syn_E': 2.5,
                   'tau_syn_I': 10.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -59
}

NetworkInfo = scipy.io.loadmat('trained weight.mat')
Fire_rate = NetworkInfo['firing_rate'][0].tolist()
weights_import = NetworkInfo['weight']
fire_table = NetworkInfo['firing_table'][0]

#print fire_table[0]
#np.array_equal(fire_table[2][0],np.array([ 11.,  23.,  24.,  30.,  32.,  34.,  37.]))
#--------------------------------------------------#
#                 Bluetooth Setup
#--------------------------------------------------#
bd_addr = "20:15:03:19:14:55" #itade address
port = 1
sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )


def convert_weights_to_list(matrix, delay):
    def build_list(indices):
        # Extract weights from matrix using indices
        weights = matrix[indices]
        # Build np array of delays
        delays = np.repeat(delay, len(weights))
        # Zip x-y coordinates of non-zero weights with weights and delays
        return zip(indices[0], indices[1], weights, delays)

    # Get indices of non-nan i.e. connected weights
    connected_indices = np.where(~np.isnan(matrix))
    # Return connection lists
    return build_list(connected_indices)

def Plot_WeightDistribution(weight,bin_num,title):
    hist,bins = np.histogram(weight,bins=bin_num)
    center = (bins[:-1]+bins[1:])/2
    width = (bins[1]-bins[0])*0.7
    plt.bar(center,hist,align='center',width =width)
    plt.xlabel('Weight')
    plt.title(title)
    plt.show()
#Plot_WeightDistribution(weights_import,200,'trained weight')

sim.setup(timestep=1, min_delay=1, max_delay=144)
synapses_to_spike = 1
delay = 2
prepop_size = 256
postpop_size = 40
animation_time = 200
episode = 200
order = np.array([0,1,2,3])
simtime = len(order)*animation_time

def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0])/1000 #from ns to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts

def ReadSpikeTime(NeuronID,x,y,ts,p,ONOFF):
    timeTuple=[]
    for idx in range(0,len(x)):
        if NeuronID == (x[idx]+y[idx]*16) and p[idx]==ONOFF:
           timeTuple.append(ts[idx])
    return timeTuple

def BuildSpike(x,y,ts,p,ONOFF):
    SpikeTimes = []
    for i in range(0,prepop_size):
        SpikeTimes.append(ReadSpikeTime(i,x,y,ts,p,ONOFF))
    return SpikeTimes

def BuildTrainingSpike(order,ONOFF):
    complete_Time = []
    for nid in range(0,prepop_size):
        SpikeTimes = []
        for tid in range(0,len(order)):
                temp=[]
                loc = order[tid]
                j = np.repeat(nid,len(argw[loc][1]))
                p = np.repeat(ONOFF,len(argw[loc][1]))
                temp = 200*tid+argw[loc][3][(j%16==argw[loc][0])&(j/16==argw[loc][1])&(p==argw[loc][2])]
                if temp.size>0:
                   SpikeTimes = np.concatenate((SpikeTimes,temp))
        if type(SpikeTimes) is not list:
           complete_Time.append(SpikeTimes.tolist())
        else:
            complete_Time.append([])
    return complete_Time

def plot_spikes(spikes, title):
    if spikes is not None:
        pylab.figure()
        pylab.xlim((0, simtime))
        pylab.ylim((0, postpop_size+2))
        lines = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes],"|")
        pylab.setp(lines,markersize=10,color='r')
        pylab.xlabel('Time/ms')
        pylab.ylabel('Neruons #')
        pylab.title(title)
        pylab.show()

    else:
        print "No spikes received"



(x_r, y_r, p_r, ts_r) = get_data('Data/D.mat/200ms_16_16_ws_norm_l2r.mat')
(x_l, y_l, p_l, ts_l) = get_data('Data/D.mat/200ms_16_16_ws_norm_r2l.mat')
(x_d, y_d, p_d, ts_d) = get_data('Data/D.mat/200ms_16_16_ws_norm_t2b.mat')
(x_u, y_u, p_u, ts_u) = get_data('Data/D.mat/200ms_16_16_ws_norm_b2t.mat')
argw =((x_r, y_r, p_r, ts_r),(x_l, y_l, p_l, ts_l), (x_d, y_d, p_d, ts_d),(x_u, y_u, p_u, ts_u))

#Let us only use the ON events
TrianSpikeON = BuildTrainingSpike(order,1)
#SpikeTimesOn= BuildSpike(x_r,y_r,ts_r,p_r,1)
spikeArrayOn = {'spike_times': TrianSpikeON}

ON_pop = sim.Population(prepop_size, sim.SpikeSourceArray, spikeArrayOn, label='inputSpikes_On')
post_pop= sim.Population(postpop_size,sim.IF_curr_exp, cell_params_lif, label='post_1')

connectionsOn = sim.Projection(ON_pop, post_pop, sim.FromListConnector(convert_weights_to_list(weights_import, delay)))
#inhibitory between the neurons
connection_I  = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(weights = 0.08,delays=1), target='inhibitory')
post_pop.record()
sim.run(simtime)

# == Get the Simulated Data =================================================
post_spikes = post_pop.getSpikes(compatible_output=True)
#weights_trained = connectionsOn.getWeights(format='list')

plot_spikes(post_spikes,'trained spike')
sim.end()

def GetFiringPattern(spike,low,high):
    spikeT = np.transpose(spike)
    time_stamp = spikeT[1]
    target_index = ((time_stamp-low)>=0) & ((time_stamp-high)<0)
    firingTable = np.unique(spikeT[0][target_index])
    firingRate = len(np.unique(spikeT[0][target_index]))
    return firingRate,firingTable

print Fire_rate
#test of the function Getting Fire Pattern
for jj in range(0,4):
    print GetFiringPattern(post_spikes,200*jj,200*(jj+1))
    
def CompareFiringPattern(firing_table,timeid):
    MOV =4
    rate,temp_table = GetFiringPattern(post_spikes,200*timeid,200*(timeid+1))
    for kk in range(0,4):
        if ( np.size(firing_table[3-kk]) != 0):
            #print firing_table[3-kk][0], temp_table,np.array_equal(firing_table[3-kk][0],temp_table)
            if(np.array_equal(firing_table[3-kk][0],temp_table)):
                MOV = kk
    return MOV
    
print CompareFiringPattern(fire_table,2)

'''
#left and right movement first
sock.connect((bd_addr, port))
print 'Connected'
sock.settimeout(10.0) #time scale in second

if (CompareFiringPattern==0):
    sock.send("r")
    print 'Sent right move'

if (CompareFiringPattern==1):
    sock.send("l")
    print 'Sent Left move'

if (CompareFiringPattern==2):
    sock.send("t")
    print 'Sent Forward move'
if (CompareFiringPattern==3):
    sock.send("d")
    print 'Sent Back Move'

sock.close()
'''










