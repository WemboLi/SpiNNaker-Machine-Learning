__author__ = 'wenbo'
import spynnaker.pyNN as sim
import pylab
import scipy.io
import numpy as np

import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
#--------------------------------------------------#
# System Setup and Neuron parameters Specification#
#--------------------------------------------------#
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

sim.setup(timestep=1, min_delay=1, max_delay=144)
synapses_to_spike = 1
delay = 2
prepop_size = 256
postpop_size = 40
animation_time = 200
episode = 200
order = np.concatenate((np.random.randint(0,4,episode),np.array([0,1,2,3])))
simtime = len(order)*animation_time

#------------------------------------------------#
# Building of Spikes on Virutal Retina Population#
#------------------------------------------------#
def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0])/1000 #from ns to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts

(x_r, y_r, p_r, ts_r) = get_data('Data/D.mat/200ms_16_16_ws_norm_l2r.mat')
(x_l, y_l, p_l, ts_l) = get_data('Data/D.mat/200ms_16_16_ws_norm_r2l.mat')
(x_d, y_d, p_d, ts_d) = get_data('Data/D.mat/200ms_16_16_ws_norm_t2b.mat')
(x_u, y_u, p_u, ts_u) = get_data('Data/D.mat/200ms_16_16_ws_norm_b2t.mat')

argw =((x_r, y_r, p_r, ts_r),(x_l, y_l, p_l, ts_l),
       (x_d, y_d, p_d, ts_d),(x_u, y_u, p_u, ts_u))

def raster_plot():
    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('spikes')
    for ii in range(0,len(argw)):
        pylab.plot(argw[ii][3]+200*ii,argw[ii][0]+argw[ii][1]*16,".")
    pylab.title('raster plot of Virtual Retina Neuron Population')
    pylab.show()

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
                temp = 200*tid+argw[loc][3][(j%16==argw[loc][0])&
                                            (j/16==argw[loc][1])&(p==argw[loc][2])]
                if temp.size>0:
                   SpikeTimes = np.concatenate((SpikeTimes,temp))
        if type(SpikeTimes) is not list:
           complete_Time.append(SpikeTimes.tolist())
        else:
            complete_Time.append([])
    return complete_Time
#raster plot of the firing time of the neurons

TrianSpikeON = BuildTrainingSpike(order,1)
FourDirectionTime = BuildTrainingSpike(order,1)
spikeArrayOn = {'spike_times': TrianSpikeON}
ON_pop = sim.Population(prepop_size, sim.SpikeSourceArray,
                        spikeArrayOn, label='inputSpikes_On')
post_pop= sim.Population(postpop_size,sim.IF_curr_exp,
                         cell_params_lif, label='post_1')
#------------------------------------------------#
# STDP and Neuron Network Specification#
#------------------------------------------------#
stdp_model = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(tau_plus=15.0,
                                                    tau_minus=25.0
                                                  ,nearest=True),
                weight_dependence=sim.AdditiveWeightDependence(w_min=0,
                                                               w_max=0.5,
                                                               A_plus=0.012,
                                                               A_minus=0.01))
weight_ini = np.random.normal(0.108, 0.003,prepop_size*postpop_size).tolist()
delay_ini = np.random.normal(2, 0.3,prepop_size*postpop_size).tolist()

connectionsOn = sim.Projection(ON_pop, post_pop, sim.AllToAllConnector(
    weights = weight_ini,delays=2),
                               synapse_dynamics=sim.SynapseDynamics(
                                   slow=stdp_model))
#inhibitory between the neurons
connection_I  = sim.Projection(post_pop, post_pop,
                               sim.AllToAllConnector(weights = 0.08,delays=1),
                               target='inhibitory')

post_pop.record()
#post_pop.record_v()
sim.run(simtime)

v = None
post_spikes = None
synapses =None

# == Plot the Simulated Data =================================================
post_spikes = post_pop.getSpikes(compatible_output=True)
#v = post_pop.get_v(compatible_output=True)
weights_trained = connectionsOn.getWeights(format='array')
sim.end()

def plot_spikes(spikes, title):
    if spikes is not None:
        pylab.figure()
        ax = plt.subplot(111, xlabel='Time/ms', ylabel='Neruons #',
                         title=title)
        pylab.xlim((0, simtime))
        pylab.ylim((0, postpop_size+2))
        lines = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes],"|")
        pylab.setp(lines,markersize=10,color='r')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
             item.set_fontsize(20)
        pylab.show()

    else:
        print "No spikes received"

plot_spikes(post_spikes, "Spike Pattern of Post-Synaptic Population")

#-------------------------#
# post-training processing#
#-------------------------#
def GetFiringPattern(spike,low,high):
    spikeT = np.transpose(spike)
    time_stamp = spikeT[1]
    target_index = ((time_stamp-low)>=0) & ((time_stamp-high)<0)
    firingTable = np.unique(spikeT[0][target_index])
    firingRate = len(np.unique(spikeT[0][target_index]))
    return firingRate, firingTable

def GetSpikeRecord(spike):
    spikeT = np.transpose(spike)
    time_stamp = spikeT[1]
    target_index = ((time_stamp)>=simtime-4*animation_time)
    time = spikeT[1][target_index]-simtime+4*animation_time
    neuronNum = spikeT[0][target_index]
    return time,neuronNum

def Plot_WeightDistribution(weight,bin_num,title):
    hist,bins = np.histogram(weight,bins=bin_num)
    center = (bins[:-1]+bins[1:])/2
    width = (bins[1]-bins[0])*0.7
    ax = pylab.subplot(111,xlabel='Weight',title =title)
    plt.bar(center,hist,align='center',width =width)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.show()

firing_rate = []
firing_table=[]
for jj in range(0,4):
    rate,table = GetFiringPattern(post_spikes,
                                  simtime-200*(jj+1),simtime-200*jj)
    print table
    firing_rate.append(rate)
    firing_table.append(table)
print "firing rate of each direction:",firing_rate
firing_time,neuronNum = GetSpikeRecord(post_spikes)

#save training result
scipy.io.savemat('trained_weight.mat',{'weight':weights_trained,
                                       'firing_rate':firing_rate,
                                       'firing_table':firing_table,
                                       'firing_time':firing_time,
                                       'NeuronID':neuronNum})

#Plot membrane potential of neurons
#if v != None:
#    pylab.figure(2)
#    ticks = len(v)/postpop_size
#    pylab.xlabel('Time/ms')
#    pylab.ylabel('v')
#    pylab.title('v')
#    for pos in range(0,postpop_size,2):
#          v_for_neuron = v[ticks*pos:ticks*(pos+1)]
#          pylab.plot([i[1] for i in v_for_neuron],
#                [i[2] for i in v_for_neuron])
#    pylab.show()

#Plot Synaptic Weight Distribution
Plot_WeightDistribution(weight_ini,200,'Histogram of Initial Weight')
Plot_WeightDistribution(weights_trained,200,'Histogram of Trained Weight')
