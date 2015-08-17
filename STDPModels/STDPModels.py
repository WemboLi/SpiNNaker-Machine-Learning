__author__ = 'wenbo'
import spynnaker.pyNN as sim
import pylab
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution
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
#np.random.randint(0,4,episode)
simtime = len(order)*animation_time

def get_data(filename):
    dvs_data = scipy.io.loadmat(filename)
    ts = dvs_data['ts'][0]
    ts = (ts - ts[0])/1000 #from ns to ms
    x = dvs_data['X'][0]
    y = dvs_data['Y'][0]
    p = dvs_data['t'][0]
    return x,y,p,ts
'''
(x_r, y_r, p_r, ts_r) = get_data('Data/mat/200ms_16_16_norm_l2r.mat')
(x_l, y_l, p_l, ts_l) = get_data('Data/mat/200ms_16_16_norm_r2l.mat')
(x_d, y_d, p_d, ts_d) = get_data('Data/mat/200ms_16_16_norm_t2b.mat')
(x_u, y_u, p_u, ts_u) = get_data('Data/mat/200ms_16_16_norm_b2t.mat')
'''

(x_r, y_r, p_r, ts_r) = get_data('Data/D.mat/200ms_16_16_ws_norm_l2r.mat')
(x_l, y_l, p_l, ts_l) = get_data('Data/D.mat/200ms_16_16_ws_norm_r2l.mat')
(x_d, y_d, p_d, ts_d) = get_data('Data/D.mat/200ms_16_16_ws_norm_t2b.mat')
(x_u, y_u, p_u, ts_u) = get_data('Data/D.mat/200ms_16_16_ws_norm_b2t.mat')

argw =((x_r, y_r, p_r, ts_r),(x_l, y_l, p_l, ts_l), (x_d, y_d, p_d, ts_d),(x_u, y_u, p_u, ts_u))

def Plot_WeightDistribution(weight,bin_num,title):
    hist,bins = np.histogram(weight,bins=bin_num)
    center = (bins[:-1]+bins[1:])/2
    width = (bins[1]-bins[0])*0.7
    plt.bar(center,hist,align='center',width =width)
    plt.xlabel('Weight')
    plt.title(title)
    plt.show()

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

TrianSpikeON = BuildTrainingSpike(order,1)
TrianSpikeOFF = BuildTrainingSpike(order,0)
#SpikeTimesOn= BuildSpike(x_r,y_r,ts_r,p_r,1)
#SpikeTimesOFF=BuildSpike(x,y,ts,p,0)
spikeArrayOn = {'spike_times': TrianSpikeON}
spikeArrayOff = {'spike_times': TrianSpikeOFF}

ON_pop = sim.Population(prepop_size, sim.SpikeSourceArray, spikeArrayOn, label='inputSpikes_On')
#OFF_pop = sim.Population(prepop_size, sim.SpikeSourceArray, spikeArrayOff, label='inputSpikes_Off')
post_pop= sim.Population(postpop_size,sim.IF_curr_exp, cell_params_lif, label='post_1')
stdp_model = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(tau_plus=15.0, tau_minus=25.0
                                                  ,nearest=True),
                weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.5,
                                                               A_plus=0.01, A_minus=0.01))
               # dendritic_delay_fraction=0)
weight_ini = np.random.normal(0.108, 0.003,prepop_size*postpop_size).tolist()
delay_ini = np.random.normal(2, 0.3,prepop_size*postpop_size).tolist()

connectionsOn = sim.Projection(ON_pop, post_pop, sim.AllToAllConnector(weights = weight_ini,delays=2), synapse_dynamics=sim.SynapseDynamics(slow=stdp_model))
connection_I  = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(weights = 0.08,delays=1), target='inhibitory')

#inhibitory between the neurons
#connectionsOFF = sim.Projection(OFF_pop, post_pop, sim.AllToAllConnector(weight_ini,delays=2), synapse_dynamics=sim.SynapseDynamics(slow=stdp_model))
#connectionsOn = p.Projection(ON_pop,post_pop,p.FromListConnector(loopConnections))

post_pop.record()
#post_pop.record_v()
sim.run(simtime)

v = None
post_spikes = None
synapses =None

# == Get the Simulated Data =================================================
post_spikes = post_pop.getSpikes(compatible_output=True)
#v = post_pop.get_v(compatible_output=True)
#synapses = post_pop.get_gsyn(compatible_output=True)
weights_trained = connectionsOn.getWeights(format='list')

# == Get the Simulation Result of the Data  ==================================
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

    else:
        print "No spikes received"

plot_spikes(post_spikes, "Spike Pattern of Post-Synaptic Population")
pylab.show()

#raster plot of the firing time of the neurons
def raster_plot():
    pylab.figure()
    pylab.xlabel('Time/ms')
    pylab.ylabel('spikes')
    for ii in range(0,len(argw)):
        pylab.plot(argw[ii][3]+200*ii,argw[ii][0]+argw[ii][1]*16,".")
    pylab.title('raster plot of ballmovement(neuron firing pattern)')
    pylab.show()
#raster_plot()
'''

if v != None:
    pylab.figure(2)
    ticks = len(v)/postpop_size
    pylab.xlabel('Time/ms')
    pylab.ylabel('v')
    pylab.title('v')
    for pos in range(0,postpop_size,2):
          v_for_neuron = v[ticks*pos:ticks*(pos+1)]
          pylab.plot([i[1] for i in v_for_neuron],
                [i[2] for i in v_for_neuron])
    pylab.show()


'''
print "Weights in the Network are: \n"

#-------------------------------------------------|
#calculate statistical features of the network    |
#-------------------------------------------------|

#Plot_WeightDistribution(weight_ini,200,'Histogram of Initial Weight')
#Plot_WeightDistribution(weights_trained,200,'Histogram of Trained Weight')

#print "Average spike times of each neuron:%f "% post_pop.meanSpikeCount(self, gather=True)




