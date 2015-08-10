__author__ = 'wenbo'
import spynnaker.pyNN as sim
import pylab
import scipy.io
import numpy as np
cell_params_lif = {
                   'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 10.0,
                   'tau_refrac': 5.0,
                   'tau_syn_E': 3.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
}

sim.setup(timestep=1, min_delay=1, max_delay=144)
simtime = 200
synapses_to_spike = 1
delay = 1
prepop_size = 256
postpop_size = 20
dvs_data = scipy.io.loadmat('Data/mat/200ms_16_16_norm_r2l.mat')

ts = dvs_data['ts'][0]
ts = (ts - ts[0])/1000 #from ns to ms
x = dvs_data['X'][0]
y = dvs_data['Y'][0]
p = dvs_data['t'][0]
NeuronId = x*16+y

'''
loopConnections = list()
for i in range(0, 5):
    singleConnection = (i, i+1, synapses_to_spike, delay)
    loopConnections.append(singleConnection)
    '''

def ReadSpikeTime(NeuronID,x,y,ts,p):
    timeTuple=[]
    for idx in range(0,len(x)):
        if NeuronID == (x[idx]*16+y[idx]) and p[idx]==0:
           timeTuple.append(ts[idx])
    return timeTuple

def BuildSpike(x,y,ts,p):
    SpikeTimes = []
    for i in range(0,prepop_size):
        SpikeTimes.append(ReadSpikeTime(i,x,y,ts,p))
    return SpikeTimes

SpikeTimes=BuildSpike(x,y,ts,p)
spikeArray = {'spike_times': SpikeTimes}
#spikeArray = {'spike_times': [[],[1,6,10],[],[],[40,60,80]]}

pre_pop = sim.Population(prepop_size, sim.SpikeSourceArray, spikeArray, label='inputSpikes_1')
post_pop= sim.Population(postpop_size,sim.IF_curr_exp, cell_params_lif, label='post_1')
stdp_model = sim.STDPMechanism(
                timing_dependence=sim.SpikePairRule(tau_plus=15.0, tau_minus=25.0
                                                  ,nearest=True),
                weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.05,
                                                               A_plus=0.0001, A_minus=0.00012))
                #weight=pyNN.random.RandomDistribution('normal', mu=0.001, sigma=0.001))

               # dendritic_delay_fraction=0)
weight_ini = np.random.normal(0.0475, 0.003,prepop_size*postpop_size).tolist()
connections = sim.Projection(pre_pop, post_pop, sim.AllToAllConnector(weight_ini,delays=2), synapse_dynamics=sim.SynapseDynamics(slow=stdp_model))

#connections = p.Projection(pre_pop,post_pop,p.FromListConnector(loopConnections))
#inhibitory between the neurons
#pre_pop.record('spikes')
post_pop.record()
post_pop.record_v()
post_pop.record_gsyn()
sim.run(200)

#Test the Spike Shape of the Neuron
#postsynaptic_data = post_pop.get_data().segments[0]
v = None
post_spikes = None
synapses =None

# == Get the Simulated Data =================================================
v = post_pop.get_v(compatible_output=True)
post_spikes = post_pop.getSpikes(compatible_output=True)
synapses = post_pop.get_gsyn(compatible_output=True)
weights = connections.getWeights(format='array')

# == Get the Simulation Result of the Data  ==================================
def plot_spikes(spikes, title):
    if spikes is not None:
        pylab.figure()
        pylab.xlim((0, simtime))
       # pylab.ylim((0, 4))
        lines = pylab.plot([i[1] for i in spikes], [i[0] for i in spikes],"|")
        pylab.setp(lines,linewidth=5,color='r')
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title(title)

    else:
        print "No spikes received"

plot_spikes(post_spikes, "post-synaptic")
pylab.show()

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

#raster plot of the firing time of the neurons
'''
pylab.figure()
pylab.xlabel('Time/ms')
pylab.ylabel('spikes')
pylab.title('raster plot of right to left fire neurons')
pylab.plot(ts,NeuronId, ".")
pylab.show()

#synapses of the neurons, physical meaning, not the weight?
if synapses!= None:
    pylab.figure(3)
    ticks = len(synapses)/postpop_size
    pylab.xlabel('Time/ms')
    pylab.ylabel('synapses value')
    pylab.title('synapses of the Neuron Network')
    for pos in range(0,postpop_size):
         synapses_for_neuron = synapses[ticks*pos:ticks*(pos+1)]
         pylab.plot([i[1] for i in synapses_for_neuron],
                [i[2] for i in synapses_for_neuron])
    pylab.show()
'''
print "Weights in the Network are: \n"
#print weights
#-------------------------------------------------|
#calculate statistical features of the network    |
#-------------------------------------------------|

#print "Average spike times of each neuron:%f "% post_pop.meanSpikeCount(self, gather=True)
#weightHistogram(self, min=None, max=None, nbins=10)



