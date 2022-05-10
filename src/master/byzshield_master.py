from .utils import * 
from .baseline_master import SyncReplicasMaster_NN

import logging # ~ for comments on logging see "distributed_nn.py"
import torch.optim as optim

from joblib import Parallel, delayed
from functools import reduce

# ~ for clique detection
import networkx as nx
from itertools import product
from collections import Counter

import gc

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ~ test
# writes class variables of "obj" to logger
def writeVarsLog(obj, obj_name):
    logger.info("vars("+obj_name+")")
    objVars = vars(obj) # ~ dictionary with class variables
    for name in objVars:
        var = objVars[name]
        logger.info("\n{}".format(name))
        writeVarSpecs(var, "")


# ~ test 
# goes deep into each dictionary, list, np.ndarray and prints only necessary information in an indented fashion 
def writeVarSpecs(var, indent: str):
    if isinstance(var, dict):
        logger.info(indent+"type(var):{}, len(var): {}".format(type(var), len(var)))
        for key in var:
            logger.info(indent+"\t"+"key: {}".format(key))
            writeVarSpecs(var[key], indent+"\t")
    elif isinstance(var, list):
        logger.info(indent+"type(var):{}, len(var): {}".format(type(var), len(var)))
        for elem in var:
            writeVarSpecs(elem, indent+"\t")
    elif isinstance(var, np.ndarray):
        logger.info(indent+"{} {} {}".format(type(var), var.dtype, var.shape))
    else:
        logger.info(indent+"{}".format(var))

# ~ DEBUG: this will traceback numpy warnings and crash instead of just warning
warnings.simplefilter('error')

class ByzshieldMaster(SyncReplicasMaster_NN): # ~ check if you can make it subclass of DracoLiteMaster
    def __init__(self, comm, **kwargs):
        '''master node here, no rank needed since the rank will always be 0 for master node'''
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']

        self._num_grad_to_collect = self.world_size - 1
        self.num_workers = self.world_size-1
        # used to aggregate tmp gradients, the length is the same as # of fc layer 
        self._grad_aggregate_buffer = []
        self._coded_grads_buffer = {} # ~ dict with key the file 0...f-1 and value: list (one element per worker in group that has it) of lists (each of those lists is a deep copy of self._grad_aggregate_buffer)
        self._model_shapes = []
        # self._first_grad_received = False # ~ never used
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._update_mode = kwargs['update_mode']
        self._max_steps = kwargs['max_steps'] # ~ max-steps argument of run_pytorch
        self._group_list = kwargs['group_list'] # ~ dictionary from file 0...f-1 to list of workers (ranks) that have it
        self._compress_grad = kwargs['compress_grad']
        self._checkpoint_step = kwargs['checkpoint_step']
        self._group_size = len(self._group_list[0]) # ~ == r
        self._bucket_size = kwargs['bucket_size']
        self._lis_simulation = kwargs['lis_simulation']
        self._s = kwargs['worker_fail'] # ~ == q
        self._device = kwargs['device']

        ######## LR scheduling related ############
        self.gamma = kwargs['gamma']
        self.lr_step = kwargs['lr_step']
        # self.gamma = 0.99
        # self.gamma = 0.98
        # self.lr_step = 100000000000000000
        # self.lr_step = 2*self._eval_freq
        # self.lr_step = 10
        ###########################################
        
        self.workerFileHt = kwargs['group_num'] # ~ master needs to know the files of all workers, this argument is DIFFERENT (dictionary from worker (rank) to list of files) than "group_num" (list of files for caller worker (rank)) at the worker level
        
        self.ell = len(self.workerFileHt[1]) # ~ computation load per worker ("l" in paper), assumes symmetric scheme so we pull it from worker 1
        
        self._fail_workers = kwargs['adversaries'] # ~ the same set of Byzantines will be used in ALIE (not a new random one)
        
        self._c_q_max = kwargs['c_q_max'] # ~ maximum no. of distorted files after majority voting (currently hard-coded)
        
        self._err_mode = kwargs['err_mode']
        
        self._adversarial_detection = kwargs['adversarial_detection'] # ~ method used by the PS to detect adversaries
        
        self._approach = kwargs['approach']
        
        # ~ Ranks of all workers
        self.all_workers = set(range(1, self.world_size))
        
        self._err_choice = kwargs['err_choice'] # ~ (needed here for ALIE/FoE) if set to 'fixed_disagreement', the adversaries will only distort files of groups where all honest workers are in a fixed disagreement set. If set to 'all', the adversaries will distort all files.
        
        # ~ Files in which all participants are adversaries (used if detection is enabled)
        self.fullyDistortedFiles = []
        
        self._lr_warmup = kwargs['lr_warmup']
        self._maxlr = kwargs['maxlr']
        self._maxlr_steps = kwargs['maxlr_steps']
        self._lr_annealing = kwargs['lr_annealing']
        self._lr_annealing_minlr = kwargs['lr_annealing_minlr']
        self._lr_annealing_cycle_steps = kwargs['lr_annealing_cycle_steps']
        self._max_grad_l2norm = kwargs['max_grad_l2norm']
        self._pair_groups = kwargs['pair_groups']
        self._det_win_length = kwargs['det_win_length']
        self._permute_files = kwargs['permute_files']
        self._tolerance = kwargs['tolerance']
        
        # ~ If learning rate warmup is enabled, the starting learning rate depends on that schedule
        if self._lr_warmup == "yes":
            self.lr = 1/self._maxlr_steps*self._maxlr
            
        # ~ Only used in self._attempt_degree_detection(), see there.
        self._agreements = Counter()
        if self._adversarial_detection == "degree" and self._checkpoint_step != 0:
            # In case self._checkpoint_step != 0 and if the breakpoint does not align with the start of a detection window, more than q adversaries can be detection and detection will be messed up.
            # The solution is to initialize as if all pairs agreed with each other since the start of the window.
            K = self.num_workers
            
            windowIterCtr = (self._checkpoint_step-1)%self._det_win_length + 1
            
            for worker1 in range(1, K+1):
                for worker2 in range(worker1+1, K+1):
                    self._agreements[(worker1, worker2)] = windowIterCtr*self._pair_groups
            
            # ~ This is needed for the same reason after resuming from a breakpoint. 
            # Note that in _attempt_degree_detection(), this is reset only at the start of each window at a local scope not visible in the rest of the code before it is set.
            self.detectedWorkers = []
        
        
        # ~ test
        logger.info("DEBUG_PS_BYZ: self._group_list: {}".format(self._group_list))
        # logger.info("DEBUG_PS_BYZ: self.workerFileHt: {}".format(self.workerFileHt))
        # logger.info("DEBUG_PS_BYZ: self._adversarial_detection: {}".format(self._adversarial_detection))
        # logger.info("DEBUG_PS_BYZ: self._approach: {}".format(self._approach))
        # logger.info("DEBUG_PS_BYZ: self.num_workers: {}".format(self.num_workers))
        # logger.info("DEBUG_PS_BYZ: self._group_size: {}".format(self._group_size))
        # logger.info("DEBUG_PS_BYZ: self._lr_warmup: {}".format(self._lr_warmup))
        # logger.info("DEBUG_PS_BYZ: self._maxlr: {}".format(self._maxlr))
        # logger.info("DEBUG_PS_BYZ: self._maxlr_steps: {}".format(self._maxlr_steps))
        # logger.info("DEBUG_PS_BYZ: self._lr_annealing: {}".format(self._lr_annealing))
        # logger.info("DEBUG_PS_BYZ: self._lr_annealing_minlr: {}".format(self._lr_annealing_minlr))
        # logger.info("DEBUG_PS_BYZ: self._lr_annealing_cycle_steps: {}".format(self._lr_annealing_cycle_steps))
        # logger.info("DEBUG_PS_BYZ: self._max_grad_l2norm: {}".format(self._max_grad_l2norm))
        logger.info("DEBUG_PS_BYZ: self.lr: {}".format(self.lr))
        logger.info("DEBUG_PS_BYZ: self._pair_groups: {}".format(self._pair_groups))
        logger.info("DEBUG_PS_BYZ: self._det_win_length: {}".format(self._det_win_length))
        logger.info("DEBUG_PS_BYZ: self._permute_files: {}".format(self._permute_files))
        # logger.info("DEBUG_PS_BYZ: self._tolerance: {}".format(self._tolerance))
        logger.info("DEBUG_PS_BYZ: Torch device: {} {}".format(self._device, type(self._device)))

    def build_model(self):
        # ~ test
        # torch.manual_seed(428)
        
        # build network
        if self.network_config == "LeNet":
            self.network=LeNet()
        elif self.network_config == "ResNet18":
            self.network=ResNet18()
        elif self.network_config == "ResNet34":
            self.network=ResNet34()
        elif self.network_config == "ResNet50":
            self.network=ResNet50()
        elif self.network_config == "FC":
            self.network=FC_NN()
        elif self.network_config == "DenseNet":
            self.network=DenseNet121()
        elif self.network_config == "VGG11":
            self.network=vgg11_bn(num_classes=100)
        elif self.network_config == "VGG13":
            self.network=vgg13_bn(num_classes=100)
        elif self.network_config == "VGG19":
            self.network=vgg19_bn(num_classes=100)
            
        if self._checkpoint_step != 0:
            file_path = self._train_dir + "model_step_" + str(self._checkpoint_step)
            self._load_model(file_path)
            self.cur_step = int(self._checkpoint_step)+1

        # assign a gradient accumulator to collect gradients from workers
        # self.grad_accumulator = ByzShieldGradientAccumulator(self.network, self.world_size-1, self.ell, mode=self._compress_grad) # ~ passes the number of workers as the number of gradients to accumulate and the model so that it knows the dimensions

        # ~ test
        # This changes depending on whether "compress-grad" is disabled (it could be np.ndarray or bytearray)
        # if type(self.grad_accumulator.gradient_aggregator[0][0]) == np.ndarray:
            # logger.info("PS_BYZ: Master node self.grad_accumulator.gradient_aggregator dtype: {}".format(self.grad_accumulator.gradient_aggregator[0][0].dtype))
            # logger.info("PS_BYZ: Master node self.grad_accumulator.gradient_aggregator bytes: {}".format(sum([sum([workerGrad.nbytes for workerGrad in layerLst]) for layerLst in self.grad_accumulator.gradient_aggregator])))
        # elif type(self.grad_accumulator.gradient_aggregator[0][0]) == bytearray:
            # logger.info("PS_BYZ: Master node self.grad_accumulator.gradient_aggregator bytes: {}".format(sum([sum([len(workerGrad) for workerGrad in layerLst]) for layerLst in self.grad_accumulator.gradient_aggregator])))

        self.init_model_shapes()

        # ~ test
        # logger.info("PS_BYZ: Master node self._coded_grads_buffer bytes: {}".format(sum([sum([sum([sum([g.nbytes for g in workerFileLayerGrad]) for workerFileLayerGrad in workerFileGrad]) for workerFileGrad in fileGrads]) for fileGrads in self._coded_grads_buffer.values()])))
        # logger.info("PS_BYZ: Master node self._draco_lite_aggregation_buffer bytes: {}".format(self._draco_lite_aggregation_buffer.nbytes))
        # logger.info("PS_BYZ: Master node self._grad_aggregate_buffer bytes: {}".format(sum([x.nbytes for x in self._grad_aggregate_buffer])))

        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        #self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)

        self.network.to(self._device)

    def init_model_shapes(self):
        self.tmp_aggregate_buffer = [] # ~ list (one element per model layer) of np.ndarrays of layer shapes (will be replicated f*r times, i.e., one time for each replica of each file for INPUT to 1st stage)
        # tmp_recv_buffer = [] # ~ same as tmp_aggregate_buffer but the 0-th dimension of each layer is scaled by ell to store all files' gradients of a worker's transmission, will be replicated K times instead of f*r
        self._model_param_counter = 0
        for param_idx, param in enumerate(self.network.parameters()):
            shape = param.size()
            num_params = reduce((lambda x, y: x * y), shape)
            self._model_param_counter += num_params

            all_files_shape = (shape[0]*self.ell,)+shape[1:] # ~ for all ByzShield files
            self._model_shapes.append(all_files_shape)
            # self._grad_aggregate_buffer.append(np.zeros(shape, dtype=float_type))
            self.tmp_aggregate_buffer.append(np.zeros(shape, dtype=float_type))
            # tmp_recv_buffer.append(np.zeros(all_files_shape, dtype=float_type))

        #if self._update_mode == "maj_vote" or self._update_mode == "draco_lite":
        # for k, v in self._group_list.items(): # ~ k=0...f-1, v: list of workers (ranks) with k
            # for i, l in enumerate(v): # ~ for each worker (rank) in current group v, enumeration is not needed, remove it
                # if k not in self._coded_grads_buffer.keys():
                    # self._coded_grads_buffer[k] = [copy.deepcopy(tmp_aggregate_buffer)]
                # else:
                    # self._coded_grads_buffer[k].append(copy.deepcopy(tmp_aggregate_buffer))

        # buffer setted up for draco-lite aggregations
        self._sub_grad_size = len(self._group_list) # ~ (== f) no. of groups, i.e., no. of majority votes to be aggregated at the next stage
        # self._draco_lite_aggregation_buffer = np.zeros((self._sub_grad_size, self._model_param_counter), dtype=float_type) # ~ np.ndarray of shape f x (no. of model parameters) to store the majority votes after aggregation of each file ("key" of self._coded_grads_buffer)
    
    def init_grad_accumulator(self):
        # ~ Remove the data structure of 2 steps back, i.e., the input to the 2nd filtering stage
        # Need to set to None since it is a np.ndarray
        self._draco_lite_aggregation_buffer = None
        
        self.grad_accumulator = ByzShieldGradientAccumulator(self.network, self.world_size-1, self.ell, mode=self._compress_grad)
        
    def init_coded_grads_buffer(self):
        # ~ Remove the data structure of 2 steps back, i.e., the output of the 2nd filtering stage (single aggregated gradient used for model update)
        del self._grad_aggregate_buffer
        gc.collect()
        
        self._coded_grads_buffer = {}
        for k, v in self._group_list.items():
            for i, l in enumerate(v):
                if k not in self._coded_grads_buffer.keys():
                    self._coded_grads_buffer[k] = [copy.deepcopy(self.tmp_aggregate_buffer)]
                else:
                    self._coded_grads_buffer[k].append(copy.deepcopy(self.tmp_aggregate_buffer))
                    
    def init_draco_lite_aggregation_buffer(self):
        # ~ Remove the data structure of 2 steps back, i.e., the ByzShieldGradientAccumulator to store received gradients
        del self.grad_accumulator
        gc.collect()
        
        self._draco_lite_aggregation_buffer = np.zeros((self._sub_grad_size, self._model_param_counter), dtype=float_type)
        
    def init_grad_aggregate_buffer(self):
        # ~ Remove the data structure of 2 steps back, i.e., the input to the 1st filtering stage
        del self._coded_grads_buffer
        gc.collect()
        
        self._grad_aggregate_buffer = []
        for param_idx, param in enumerate(self.network.parameters()):
            shape = param.size()
            self._grad_aggregate_buffer.append(np.zeros(shape, dtype=float_type))
            
        # logger.info("DEBUG_PS_BYZ: len(_grad_aggregate_buffer) of master: {}".format(len(self._grad_aggregate_buffer)))
        # for i in range(len(self._grad_aggregate_buffer)):
            # logger.info("DEBUG_PS_BYZ: _grad_aggregate_buffer[i] of master: {}, {}".format(type(self._grad_aggregate_buffer[i]), self._grad_aggregate_buffer[i].shape))
        

    def start(self):
        # ~ test
        # Note: this is useless if you use init_grad_aggregate_buffer()
        # logger.info("DEBUG_PS_BYZ: len(_grad_aggregate_buffer) of master: {}".format(len(self._grad_aggregate_buffer)))
        # for i in range(len(self._grad_aggregate_buffer)):
            # logger.info("DEBUG_PS_BYZ: _grad_aggregate_buffer[i] of master: {}, {}".format(type(self._grad_aggregate_buffer[i]), self._grad_aggregate_buffer[i].shape))
            
        # ~ test
        # for x in self.network.parameters():
            # logger.info("DEBUG_PS_BYZ: network.parameters()[i] of master: {}, {}, {}".format(type(x), x.shape, x.size()))
        
        # ~ test
        # assert 0 == 1
        
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure value fetched from ps is 1
        self.async_bcast_step() # ~ (baseline) updates step and sends it to workers, matches worker's sync_fetch_step()

        # fake test here:
        for i in range(self._checkpoint_step+1, self._max_steps+1): # ~ global steps loop across all epochs
            # switch back to training mode
            self.network.train() # ~ this only sets the model to training mode, it does not actually train anything
            # self._first_grad_received = False # ~ never used
            enough_gradients_received = False

            logger.info("PS_BYZ: Master node is entering step: {}".format(i))
            self.async_bcast_step() # ~ see above...
            
            # ~ test
            # logger.info("DEBUG_PS_BYZ: Boss finished async_bcast_step for step {}".format(i))

            self.async_bcast_layer_weights_bcast()
            
            self.init_grad_accumulator()
            self.init_coded_grads_buffer()
            
            # set the gradient fetch step and gather the request
            gradient_fetch_requests=self.async_fetch_gradient_start()
            # wait for enough gradients to be aggregated:
            while not enough_gradients_received: # ~ wait for any one worker to send a gradient
                status = MPI.Status()
                if self._compress_grad == "None": # ~ received_msg: np.ndarray
                    MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)
                elif self._compress_grad == "compress": # ~ received_msg: received Python object, first (unassigned) value "_" is the MPI index of handle that completed (integer), here in [0,1,...,K*(no. of layers)-1] but no need to use it since the requests are associated with worker ranks/layer indices
                    _, received_msg=MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                    received_grad=decompress(received_msg)

                if status.tag-88 in self.grad_accumulator.model_index_range: # ~ checks if received layer index is within bounds of the list
                    # if not self._first_grad_received: # ~ never used
                        # self._first_grad_received=True
                        # grad_gather_start_time = time.time()

                    layer_index = status.tag-88

                    if self._compress_grad == "None":
                        received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
                    # do gradient shape check here
                    assert (received_grad.shape == self._model_shapes[layer_index])

                    # ~ test
                    # if status.source == 5: # ~ worker with this rank
                        # if layer_index == 1:
                            # logger.info("DEBUG_PS_BYZ: received_grad from {} for layer {}: {} {} {}".format(status.source, layer_index, type(received_grad), received_grad.shape, received_grad))
                                
                                
                    # aggregate the gradient
                    if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._num_grad_to_collect:
                        self.aggregate_gradient(received_grad, layer_index, status.source)

                    self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
                
                # Check if for each layer you have received some gradient from every worker
                enough_gradients_received = True
                for j in self.grad_accumulator.gradient_aggregate_counter:
                    enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)


            ################### "A Little is enough" attack simulation on the PS"#########################
            if self._lis_simulation == "simulate":
                self._LIE_attack_simulation()
            elif self._err_mode == "foe":
                self._FoE_attack_simulation()
            else: # ~ the rest of the attacks are simulated at the worker level
                pass  

            # ~ test
            # outputs log to file, "w+" is to overwrite existing file
            # SOS: you need to set the file handler here not while this file is imported by util.py since it will mix up logs.
            # But this handler is not needed if a file handler has been set before, e.g., in "distributed_nn.py".
            setLogFh = False
            if setLogFh:
                fh = logging.FileHandler('PS.log', 'w+')
                fh.setLevel(logging.INFO)
                logger.addHandler(fh)

            # ~ test
            # writeVarsLog(self, "PS_after_ALIE")
            
            # ~ Clip all gradients: those directly returned by the workers and those altered by ALIE or FoE
            if self._max_grad_l2norm > 0:
                self._clip_worker_gradients()
            
            method_start = time.time()
            #self._grad_aggregate_buffer = draco_lite_aggregation(self._coded_grads_buffer, self._bucket_size, self.network, self._grad_aggregate_buffer)
            self._draco_lite_aggregation()
            method_duration = time.time() - method_start

            update_start = time.time()
            # update using SGD method
            self._model_update()
            
            # ~ test
            # for layer_idx, layer in enumerate(self.network.parameters()):
                # lay = layer.detach().numpy().astype(float_type)
                # np.save('BYZSHIELD_master_model_after_step_lay'+str(layer_idx), lay)

            update_duration = time.time() - update_start
            # reset essential elements
            # self.meset_grad_buffer()
            # self.grad_accumulator.meset_everything()

            logger.info("PS_BYZ: Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration, update_duration))
            
            #for param_group in self.optimizer.param_groups:
            #    break
            logger.info("PS_BYZ: Real time Lr: {}".format([param_group['lr'] for param_group in self.optimizer.param_groups]))   

            self.cur_step += 1
            
            # PS permutes the file assignments and sends the updates to the workers
            if self._permute_files == 'yes':
                # logger.info("DEBUG_PS_BYZ: (BEFORE) PS: {} {} {} {}".format(self._group_list, type(self._group_list), self.workerFileHt, type(self.workerFileHt)))
                
                self.permuteListSets(self.num_workers)
                
                # logger.info("DEBUG_PS_BYZ: (AFTER) PS: {} {} {} {}".format(self._group_list, type(self._group_list), self.workerFileHt, type(self.workerFileHt)))

                self.sendPermToWorkers(self.num_workers)
            
            # ~ test
            # if self.cur_step == 2:
                # break

    # ~ just stores each received gradient in self._coded_grads_buffer
    # Arguments:
    # gradient: np.ndarray of shape [ell*(0-th dimension of given layer), 1-st dimension of given layer, ...]
    # layer_idx: gradient's layer index
    # source: rank of transmitter worker
    def aggregate_gradient(self, gradient, layer_idx, source):
        '''
        keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`
        '''
        #if self._update_mode == "normal":
        #    self._grad_aggregate_buffer[layer_idx] += gradient
        #elif self._update_mode == "maj_vote" or self._update_mode == "draco_lite":
        # under development, stay tunned
        
        # ~ unwrap gradient of worker and store each file's gradient to appropriate buffer
        fg0dim = self._model_shapes[layer_idx][0]//self.ell # ~ 0-th dimension for a single file
        for i, file in enumerate(self.workerFileHt[source]): # ~ for each file of "source" worker, unwrap it from received buffer
            v = self._group_list[file] # ~ group of workers with current file
            file_grad = gradient[fg0dim*i:fg0dim*(i+1), ...] # ~ gradient for a single file
            assert self._coded_grads_buffer[file][v.index(source)][layer_idx].shape == file_grad.shape
            self._coded_grads_buffer[file][v.index(source)][layer_idx] = file_grad
            
            # ~ test            
            # if self.cur_step == 2:
                # lay = 0 # ~ layer index (0-indexed)
                # # if source == 1: # ~ worker with rank ...
                # if layer_idx == lay:
                    # np.save('PS_worker'+str(source)+'_grads_layer'+str(lay)+'_file'+str(file), self._coded_grads_buffer[file][v.index(source)][layer_idx])
                        
    
    # ~ Implementation of cosine annealing based on https://www.jeremyjordan.me/nn-learning-rate/ advised by https://github.com/JonasGeiping/fullbatchtraining
    # Assumption 1: Cosine annealing takes place after the end of warmup to reach self._maxlr (or at the beginning of the training if self._maxlr_steps == 0) but not before it.
    #   So, self.cur_step-self._maxlr_steps >= 0
    # Assumption 2: Maximum learning rate is the same as self._maxlr
    def _cosine_annealing_lr(self):
        # ~ Steps elapsed since last annealing restarted
        Tcurrent = (self.cur_step-self._maxlr_steps)%self._lr_annealing_cycle_steps
        
        # ~ test
        # logger.info("DEBUG_PS_BYZ: Tcurrent: {}".format(Tcurrent))
        
        return self._lr_annealing_minlr + 0.5*(self._maxlr - self._lr_annealing_minlr)*(1 + math.cos(Tcurrent/self._lr_annealing_cycle_steps*math.pi))
        
        
    # ~ Clip gradient based on https://github.com/JonasGeiping/fullbatchtraining (advised by the same paper).
    # The norm utilized is the Euclidean norm.
    def _clip_gradient(self):
        # ~ Current value of the norm
        grad_norm = np.linalg.norm(np.stack([np.linalg.norm(g) for g in self._grad_aggregate_buffer]))
        
        # ~ test
        logger.info("DEBUG_PS_BYZ: Before clipping _grad_aggregate_buffer norm: {}".format(np.linalg.norm(np.stack([np.linalg.norm(g) for g in self._grad_aggregate_buffer]))))
        
        # ~ Clip norm using element-wise multiplication. This operation is out-of-place.
        if grad_norm > self._max_grad_l2norm:
            for i,g in enumerate(self._grad_aggregate_buffer):
                self._grad_aggregate_buffer[i] = np.multiply(g, self._max_grad_l2norm/(grad_norm + 1e-6))
            
        # ~ test
        logger.info("DEBUG_PS_BYZ: After clipping _grad_aggregate_buffer norm: {}".format(np.linalg.norm(np.stack([np.linalg.norm(g) for g in self._grad_aggregate_buffer]))))
        
        
    # ~ Clip all gradients returned by the workers based on https://github.com/JonasGeiping/fullbatchtraining
    # The method is the same as _clip_gradient(), also see comments there.
    def _clip_worker_gradients(self):
        # ~ For each file 0,...,f-1
        for fileInd in self._coded_grads_buffer:
            # ~ For each worker that processed that file, "grad" is the gradient of the entire model.
            for i,grad in enumerate(self._coded_grads_buffer[fileInd]):
                # ~ Clip gradient
                
                # ~ test
                # logger.info("DEBUG_PS_BYZ: Before clipping _coded_grads_buffer[{}][{}] norm: {}".format(np.linalg.norm(np.stack([np.linalg.norm(g) for g in grad])), fileInd, i))
        
                grad_norm = np.linalg.norm(np.stack([np.linalg.norm(g) for g in grad]))
                if grad_norm > self._max_grad_l2norm:
                    for j,g in enumerate(self._coded_grads_buffer[fileInd][i]):
                        self._coded_grads_buffer[fileInd][i][j] = np.multiply(g, self._max_grad_l2norm/(grad_norm + 1e-6))
                        
                # ~ test
                # logger.info("DEBUG_PS_BYZ: After clipping _coded_grads_buffer[{}][{}] norm: {}".format(np.linalg.norm(np.stack([np.linalg.norm(g) for g in grad])), fileInd, i))

    
    def _model_update(self):
        if self._lr_warmup == "no" and self._lr_annealing == "no":
            if self._checkpoint_step != 0 and self.cur_step == self._checkpoint_step+1:
                # ~ Apply learning rate decay when recovering from a checkpoint. This is to apply the decayed rate immediately without waiting for the condition "self.cur_step % self.lr_step == 0".
                self.optimizer.lr_update(updated_lr=(self.lr * self.gamma ** (self.cur_step // self.lr_step)))
            else:
                # ~ Apply learning rate decay as usual
                if self.cur_step % self.lr_step == 0:
                    self.optimizer.lr_update(updated_lr=(self.lr * self.gamma ** (self.cur_step // self.lr_step)))
        else:
            if self._lr_warmup == "yes" and self.cur_step <= self._maxlr_steps:
                # ~ This assumes that learning rate warmup starts at the beginning of training
                self.optimizer.lr_update(updated_lr=(self.cur_step/self._maxlr_steps)*self._maxlr)
            
            # ~ This assumes that annealing happens after the end of warmup (or at the beginning of the training if self._maxlr_steps == 0) but not before it
            if self._lr_annealing == "yes" and self.cur_step > self._maxlr_steps:
                self.optimizer.lr_update(updated_lr=self._cosine_annealing_lr())
        
        # ~ Clip gradient if a max value for the norm has been specified.
        # This will change self._grad_aggregate_buffer out-of-place.
        if self._max_grad_l2norm > 0:
            self._clip_gradient()
            
        # ~ For both BYZSHIELD and ASPIS, no need to bother as the function does the same thing in both cases.
        if self._device == torch.device("cpu"):       
            self.optimizer.step(grads=self._grad_aggregate_buffer, mode="byzshield_cpu")
        else:
            self.optimizer.step(grads=self._grad_aggregate_buffer, mode="byzshield_gpu")
            
            
    # multi-core optimized verion
    def _draco_lite_aggregation(self):
    
        # ~ test
        # lay = 0 # ~ layer index (0-indexed)
        # if self.cur_step == 7:
            # for file in self._group_list:
                # group = self._group_list[file]
                # for worker in group:
                    # for layer_idx, _ in enumerate(self.network.parameters()):
                        # if layer_idx == lay:
                            # np.save('PS_received_worker'+str(worker)+'_grads_layer'+str(lay)+'_file'+str(file), self._coded_grads_buffer[file][group.index(worker)][layer_idx])
    
        # logger.info("_draco_lite_aggregation_buffer before shape: {}".format(self._draco_lite_aggregation_buffer.shape))
        
        self.init_draco_lite_aggregation_buffer()
        
        if self._adversarial_detection == "clique":
            # ~ Try to find the adversaries using clique-finding
            self._attempt_clique_detection()
        
            if len(self.maxCliques) > 1:
                # ~ test
                # logger.info("DEBUG_PS_BYZ: DETECTION CASE 1...")
                
                # ~ Case 1: There are multiple maximum cliques so detection is infeasible, resort to the classic majority voting
                self._grad_majority_vote()
            elif len(self.detectedWorkers) < self._s:
                # ~ test
                # logger.info("DEBUG_PS_BYZ: DETECTION CASE 2...")
                
                # ~ Case 2: There is a unique maximum clique but less than q adversaries have been detected.
                # In this case, perform a "fast" majority vote, i.e., arbitrarily pick any gradient within each group since all of them have been filtered in _attempt_clique_detection() to be identical.
                # Note: Case 2 can happen if, say u adversaries agree with all K-q honest workers and with themselves. But, they can distort the (u choose r) groups in which there is no honest worker.
                #   However, even in those groups that they distort the MUST agree with each other so all group's gradients will be the same. So, "fast" majority makes sense.
                self._grad_fast_majority_vote()
            elif len(self.detectedWorkers) == self._s:
                # ~ test
                # logger.info("DEBUG_PS_BYZ: DETECTION CASE 3...")
                
                # ~ Case 3: There is a unique maximum clique and all q adversaries have been detected.
                # As in Case 2, we will arbitrarily pick one gradient within each group and then perform average across all of them (1 bucket, no median) assuming that we use "self._update_mode == coord-median".
                self._grad_fast_majority_vote()
                self._bucket_size = 1
            else:
                # ~ This should not be possible in theory; but it may be in practice.
                assert 0 == 1, "***WARNING***: More than q adversaries have been detected! Check for precision issues and increase the tolerance of np.allclose()!"
                
            # ~ Throw away all fully distorted files after detection (optional).
            # Note: it has been tested only with "coord-median"
            if len(self.fullyDistortedFiles) > 0:
                # ~ out-of-place deletion of the corresponding rows (0-th dimension)
                self._draco_lite_aggregation_buffer = np.delete(self._draco_lite_aggregation_buffer, self.fullyDistortedFiles, axis = 0)
                
                # ~ Update no. of buckets accordingly so that there are no empty slices during aggregation
                self._bucket_size = min(self._bucket_size, self._draco_lite_aggregation_buffer.shape[0])
                
                # ~ test
                # logger.info("Updated _bucket_size: {}".format(self._bucket_size))
                
        elif self._adversarial_detection == "degree":
            # ~ Try to find the adversaries using degree-based detection
            self._attempt_degree_detection()
            
            # ~ The rest is similar to clique detection above but with a difference, see below.
            if len(self.detectedWorkers) < self._s:
                # ~ Difference: we must do vanilla majority voting, not "fast" majority voting. 
                # The reason is that, unlike clique-finding, post-processing in _attempt_degree_detection() does not guarantee that all gradients within each group to be identical. 
                # Also, see the first if condition in the post-processing of the gradients after detection.
                self._grad_majority_vote()
            elif len(self.detectedWorkers) == self._s:
                self._grad_fast_majority_vote()
                self._bucket_size = 1
            else:
                # ~ This should not be possible in theory; but it may be in practice.
                assert 0 == 1, "***WARNING***: More than q adversaries have been detected! Check for precision issues and increase the tolerance of np.allclose()!"
                
                
            # ~ Same as clique detection above
            if len(self.fullyDistortedFiles) > 0:
                self._draco_lite_aggregation_buffer = np.delete(self._draco_lite_aggregation_buffer, self.fullyDistortedFiles, axis = 0)
                self._bucket_size = min(self._bucket_size, self._draco_lite_aggregation_buffer.shape[0])

        else:
            # ~ Detection is disabled, resort to the classic majority voting
            self._grad_majority_vote()
            
        # ~ test
        # lay = 0 # ~ layer index (0-indexed)
        # if self.cur_step == 2:
            # for file in self._group_list:
                # group = self._group_list[file]
                # for worker in group:
                    # for layer_idx, _ in enumerate(self.network.parameters()):
                        # if layer_idx == lay:
                            # np.save('PS_filtered_worker'+str(worker)+'_grads_layer'+str(lay)+'_file'+str(file), self._coded_grads_buffer[file][group.index(worker)][layer_idx])
        
        # ~ test
        # writeVarsLog(self, "PS_after_majority")
        # logger.info("_draco_lite_aggregation_buffer after shape: {}".format(self._draco_lite_aggregation_buffer.shape))

        # optimization objectives:
        # i) get rid of concated_grads and making it predefined
        # ii) use fancy slicing instead of the second for loop to get rid of grad_transformer (done, need to be further tested)
        # need to double check
        if self._update_mode == "coord-median":
            # Below, you need to one method, say "method x" and uncomment the corresponding lines labeled "method x"
            
            # bucketing stage
            
            # ~ 1D np.ndarray [0,...,f-1]
            # Note: the # of ignored files is subtracted
            indices = np.arange(self._sub_grad_size - len(self.fullyDistortedFiles))
            
            # ~ in-place random shuffling, changes assignment of votes to buckets
            # needs to be commented for reproducibility if you do median-of-means, for just mean or median it should have no effect
            np.random.seed()
            np.random.shuffle(indices) # ~ comment for reproducibility
            
            # ~ method 2 (see below for method 1): works for buckets of potentially unequal sizes
            grad_transformer = self._draco_lite_aggregation_buffer[indices, :]
            buckets = np.array_split(grad_transformer, self._bucket_size, axis=0) # ~ returns list, won't raise exception if the split does not result in an equal division
            grad_mean = [np.mean(x, axis=0) for x in buckets]
            aggr_res = np.median(np.array(grad_mean), axis=0)
            aggr_res = np.squeeze(aggr_res) # ~ removes extra dimension of size 1 to match shape with that of method 1
            
            # ~ test
            # logger.info("indices,  grad_transformer.shape, self.fullyDistortedFiles: {}, {}, {}".format(sorted(indices), grad_transformer.shape, self.fullyDistortedFiles))
            
            # ~ method 1 (all code below except debugging code should not be used if method 2 is uncommented)
            # splits "indices" 1D np.ndarray into self._bucket_size lists of equal-sized 1-D np.ndarrays of length f/self._bucket_size == no. of buckets.
            # Then, it concatenates them as rows to a 2D np.ndarray of shape: self._bucket_size x (f/self._bucket_size == no. of buckets)
            # random_indicies = np.array(np.split(indices, self._bucket_size))

            # ~ this assigns the rows of self._draco_lite_aggregation_buffer (1st stage majority votes, one per file) 
            # to a 3D np.ndarray of shape: (self._bucket_size) x (no. of buckets) x (total no. of "flattened" model parameters) 
            # The [i,j,:] slice is equal to self._draco_lite_aggregation_buffer[random_indicies[i,j]], i.e., 
            # it is matched with the shuffled "random_indicies", this is assigning majority votes to buckets
            # grad_transformer = self._draco_lite_aggregation_buffer[random_indicies, :]
            # num_buckets = grad_transformer.shape[0] # ~ never used

            # aggregation stage
            # ~ this should be wrong since it's aggregating across buckets (axis = 1) and then across those averages (axis = 0)
            # I think it should be axis = 0 and then axis = 1. But if you think of "--bucket-size" argument as no. of buckets, it is OK.
            
            # ~ average across dimension 1, i.e., computes mean within all buckets, 
            # output is of shape (self._bucket_size) x (total no. of "flattened" model parameters)
            # grad_mean = np.mean(grad_transformer, axis=1)
            
            # ~ test
            # np.save('BYZSHIELD_grad_mean', grad_mean)
            # if self.cur_step%self._eval_freq == 0:
                # if self._s == 0: # ~ clean
                    # np.save('BYZSHIELD_grad_mean_clean_step'+str(self.cur_step), grad_mean)
                # else: # ~ adversarial
                    # np.save('BYZSHIELD_grad_mean_byz_step'+str(self.cur_step)+'_q'+str(self._s), grad_mean)
            
            # ~ median across dimension 0, i.e., computes median of all bucket averages, 
            # output is of shape (total no. of "flattened" model parameters) x 1
            # aggr_res = np.median(grad_mean, axis=0)
            
            # ~ test
            # np.save('BYZSHIELD_aggr_res', aggr_res)
            # np.save('BYZSHIELD_grad_transformer', grad_transformer)
            # np.save('BYZSHIELD_draco_lite_aggregation_buffer', self._draco_lite_aggregation_buffer)
            # if self.cur_step%self._eval_freq == 0:
                # if self._s == 0: # ~ clean
                    # np.save('BYZSHIELD_aggr_res_clean_step'+str(self.cur_step), aggr_res)
                # else: # ~ adversarial
                    # np.save('BYZSHIELD_aggr_res_byz_step'+str(self.cur_step)+'_q'+str(self._s), aggr_res)
            
            # ~ test
            # logger.info("DEBUG_PS_BYZ: random_indicies: {} {}".format(type(random_indicies), random_indicies.shape))
            # logger.info("DEBUG_PS_BYZ: grad_transformer: {} {}".format(type(grad_transformer), grad_transformer.shape))
            # logger.info("DEBUG_PS_BYZ: grad_mean: {} {}".format(type(grad_mean), grad_mean.shape))
            # logger.info("DEBUG_PS_BYZ: aggr_res: {} {}".format(type(aggr_res), aggr_res.shape))
            
        elif self._update_mode == "sign-sgd":
            # ~ Below, you need to use one method, say "method x" and uncomment the corresponding lines labeled "method x"
            
            indices = np.arange(self._sub_grad_size)
            np.random.seed()
            np.random.shuffle(indices) # ~ comment for reproducibility

            # ~ method 2 (see coord-median for comments): works for buckets of potentially unequal sizes
            grad_transformer = self._draco_lite_aggregation_buffer[indices, :]
            buckets = np.array_split(grad_transformer, self._bucket_size, axis=0)
            tempt_aggr_res = [np.sign(np.sum(x, axis=0)) for x in buckets]
            aggr_res = np.sign(np.sum(np.array(tempt_aggr_res), axis=0))
            aggr_res = np.squeeze(aggr_res)
            
            # ~ method 1 (all code below except debugging code should not be used if method 2 is uncommented)
            # random_indicies = np.array(np.split(indices, self._bucket_size))

            #logger.info("random indicies: {}".format(random_indicies))
            # grad_transformer = self._draco_lite_aggregation_buffer[random_indicies, :]

            # aggregation stage
            # the first maj vote stage
            # tempt_aggr_res = np.sign(np.sum(grad_transformer, axis=1))
            
            # the second maj vote stage
            # aggr_res = np.sign(np.sum(tempt_aggr_res, axis=0))
            
            # ~ test
            # np.save('BYZSHIELD_tempt_aggr_res', tempt_aggr_res)
            # np.save('BYZSHIELD_aggr_res', aggr_res)
        
        elif self._update_mode == "bulyan":
            # ~ Arguments:
            # grad_list (== self._draco_lite_aggregation_buffer): np.ndarray of shape f x (no. of model parameters) to store the majority votes after aggregation of each file
            def __bulyan(grad_list, nb_in_score, bulyan_inds):
                '''
                Method introduced by: https://arxiv.org/abs/1703.02757
                '''
                neighbor_distances = [] # ~ list (of length f) of lists (each of length f-1) of floats, L-2 norm between all pairs of gradients
                for i, g_i in enumerate(grad_list): # i=0...f-1
                    
                    # ~ test
                    # logger.info("DEBUG_PS_BYZ: bulyan i, g_i: {} {} {}".format(i, type(g_i), g_i.shape))
                    
                    distance = []
                    for j in range(i+1, len(grad_list)):
                        if i != j: # ~ is this needed?
                            g_j = grad_list[j]
                            distance.append(float(np.linalg.norm(g_i-g_j)**2))
                    neighbor_distances.append(distance)
                    
                # ~ test
                # logger.info("DEBUG_PS_BYZ: neighbor_distances: {} {} {} {}".format(type(neighbor_distances), len(neighbor_distances), len(neighbor_distances[0]), neighbor_distances[0]))

                scores = []
                for i, g_i in enumerate(grad_list):
                    dists = []
                    for j, g_j in enumerate(grad_list):
                        if j == i:
                            continue
                        if j < i:
                            dists.append(neighbor_distances[j][i - j - 1])
                        else:
                            dists.append(neighbor_distances[i][j - i - 1])
                    # alternative to topk in pytorch and tensorflow
                    # ~ re-order distances s.t. the ones smaller than dists[nb_in_score] are on its left and the rest on its right, then pick the left part
                    # np.argpartition() does not sort, only does a quickselect
                    topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
                    scores.append(sum(np.take(dists, topk_ind))) # ~ np.take() simply returns the values in the specified indices

                topk_ind = np.argpartition(scores, nb_in_score)[:nb_in_score]
                selected_grads = np.array(grad_list)[topk_ind, :]

                # starting the second stage bulyan step
                # let's start to parallelize this step: the strategy is parallel([nb_in_score, d/k], ..., [nb_in_score, d/k])
                grad_dim = selected_grads.shape[1] # ~ total no. of "flattened" model parameters
                
                # ~ test
                # logger.info("DEBUG_PS_BYZ: Bulyan grad_dim: {}".format(grad_dim))

                temp_selected_grads = []
                num_pieces = 8
                segment_size = int(grad_dim / num_pieces)
                sement_counter = 0
                for i in range(num_pieces-1):
                    temp_selected_grads.append(selected_grads[:, sement_counter:sement_counter+segment_size])
                    sement_counter += segment_size
                temp_selected_grads.append(selected_grads[:, sement_counter:])

                temp_sorted_selected_grads = Parallel(n_jobs=-1, prefer="threads")(delayed(np.sort)(g, axis=0) for g in temp_selected_grads)
                sorted_selected_grads = np.concatenate(temp_sorted_selected_grads, axis=1)
                
                bulyan_selected_grads = sorted_selected_grads[bulyan_inds, :]
                aggregated_grad = np.mean(bulyan_selected_grads, axis=0)
                return aggregated_grad

            # figure out bulyan statistics
            # where `effective_s` denotes the worst case number of Byzantine workers after the majority voting stage # ~ "groups" not workers
            effective_s = self._c_q_max
            
            # ~ test
            # logger.info("DEBUG_PS_BYZ: Bulyan effective_s: {}".format(effective_s))
            
            nb_in_score = self._draco_lite_aggregation_buffer.shape[0] - effective_s - 2
            pivot = int(nb_in_score/2)
            bulyan_s = nb_in_score - 2 * effective_s

            if nb_in_score % 2 == 0:
                # even case
                bulyan_inds = [(pivot- 1 - i) for i in range(1, int((bulyan_s-2)/2)+1)] + [pivot-1, pivot] + [(pivot + i) for i in range(1, int((bulyan_s-2)/2)+1)]
            else:
                # odd case
                bulyan_inds = [(pivot - i) for i in range(1, int((bulyan_s-1)/2)+1)] + [pivot] + [(pivot + i) for i in range(1, int((bulyan_s-1)/2)+1)]

            aggr_res = __bulyan(self._draco_lite_aggregation_buffer, nb_in_score, bulyan_inds)

        elif self._update_mode == "multi-krum":
            def __krum(grad_list, s, num_workers):
                '''
                Method introduced by: https://arxiv.org/abs/1703.02757
                '''
                neighbor_distances = []
                for i, g_i in enumerate(grad_list):
                    distance = []
                    for j in range(i+1, len(grad_list)):
                        if i != j:
                            g_j = grad_list[j]
                            distance.append(float(np.linalg.norm(g_i-g_j)**2))
                    neighbor_distances.append(distance)

                # compute scores
                nb_in_score = num_workers - s - 2
                scores = []
                for i, g_i in enumerate(grad_list):
                    dists = []
                    for j, g_j in enumerate(grad_list):
                        if j == i:
                            continue
                        if j < i:
                            dists.append(neighbor_distances[j][i - j - 1])
                        else:
                            dists.append(neighbor_distances[i][j - i - 1])
                    # alternative to topk in pytorch and tensorflow
                    topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
                    scores.append(sum(np.take(dists, topk_ind)))

                topk_ind = np.argpartition(scores, nb_in_score)[:nb_in_score]
                aggregated_grad = np.mean(np.array(grad_list)[topk_ind, :], axis=0)
                return aggregated_grad

            effective_s = self._c_q_max # ~ max. no. of distorted groups
            
            # ~ test
            # logger.info("DEBUG_PS_BYZ: Multi-Krum effective_s: {}".format(effective_s))
            
            # hard coded for now
            #temp_aggr_res = []
            #for i in range(2):
            #    temp_grads = grad_transformer[i]
            #    temp = __krum(temp_grads, effective_s, temp_grads.shape[0])
            #    temp_aggr_res.append(temp)
            #aggr_res_temp = np.concatenate(temp_aggr_res, axis=0)

            # with bucketing:
            # bucketing stage in multi-krum backened draco-lite
            indices = np.arange(self._sub_grad_size)
            np.random.seed()
            np.random.shuffle(indices) # ~ comment for reproducibility

            ## Note that this part is hard coded currently
            random_indicies = np.array_split(indices, self._bucket_size)
            
            # ~ test
            # logger.info("DEBUG_PS_BYZ: random_indicies: {}".format(random_indicies))
            
            grad_transformer = np.array([self._draco_lite_aggregation_buffer[rand_inds, :] for rand_inds in random_indicies])
            aggr_res_temp = Parallel(n_jobs=-1, prefer="threads")(delayed(__krum)(grad_transformer[i], effective_s, grad_transformer[i].shape[0]) for i in range(self._bucket_size))
            
            # ~ test
            # np.save('BYZSHIELD_aggr_res_temp', aggr_res_temp)
            
            aggr_res = np.mean(np.array(aggr_res_temp), axis=0)

            # without bucketing:
            #aggr_res = __krum(self._draco_lite_aggregation_buffer, effective_s, self._draco_lite_aggregation_buffer.shape[0])

        self.init_grad_aggregate_buffer()
        
        # ~ This "unflattens" the final gradient for each model layer to match model shape for model update
        index_pointer = 0
        for j, p in enumerate(self.network.parameters()):
            grad_size = reduce((lambda x, y: x * y), p.size()) # ~ total no. of "flattened" model parameters
            drac_lite_median = aggr_res[index_pointer:index_pointer+grad_size]
            index_pointer += grad_size
            self._grad_aggregate_buffer[j] = drac_lite_median.reshape(self._grad_aggregate_buffer[j].shape)
            
            # ~ test
            # np.save('BYZSHIELD_grad_aggregate_buffer_lay'+str(j), self._grad_aggregate_buffer[j])
            
        # ~ test
        # writeVarsLog(self, "PS_after_3rd_level")


    def _grad_majority_vote(self):
        for k, v in self._coded_grads_buffer.items(): # ~ for each file
            index_pointer = 0
            for j, p in enumerate(self.network.parameters()): # ~ for each layer
                grad_size = reduce((lambda x, y: x * y), p.size()) # ~ no. of "flattened" layer dimensions
                _maj_counter = 0

                for i, elem in enumerate(v): # ~ for each worker with file k, enumeration never used
                    if _maj_counter == 0:
                        _maj_grad = elem[j] # ~ get the current layer from current worker-file gradient
                        _maj_counter = 1
                    # elif np.array_equal(elem[j], _maj_grad): # ~ Exact equality check - It does NOT work anymore due to precision issues.
                    elif self.gradientsTolEqual(elem[j], _maj_grad): # ~ tolerance-based equality check
                        _maj_counter += 1
                    else:
                        _maj_counter -= 1
                        
                    # ~ test
                    # if k == 0: # ~ k-th group (file)
                    # np.save('BYZSHIELD_coded_grads_buffer_file_'+str(k)+'_workerInGroupInd_'+str(i)+'_lay'+str(j), elem[j])
                        
                # ~ test
                # lay = 1
                # if k == 0: # ~ k-th group (file)
                # if j == lay: # ~ j-th layer
                    # np.save('BYZSHIELD_draco_lite_aggregation_buffer_file_'+str(k)+'_lay'+str(j), _maj_grad)
                        
                # ~ just a check that the gradient shape matches with the final shape for model update
                # try:
                    # assert self._grad_aggregate_buffer[j].shape == _maj_grad.shape
                # except AssertionError as e:
                    # warnings.warn("Gradient shapes incompatible, deprecated! ")

                # ~ flatten gradient (majority vote) and save it to a np.ndarray 
                # the row is indexed by file and the column is indexed by "flattened" layer dimensions
                self._draco_lite_aggregation_buffer[k, index_pointer:index_pointer+grad_size] = _maj_grad.reshape(-1)
                index_pointer += grad_size
                
                
    # ~ This function is paired with adversarial detection. It is same as _grad_majority_vote() but within each group it picks one arbitrary gradient as the output to be used in robust aggregation.
    # Note: this assumes that all r gradients of the group are identical.
    # See comments in _grad_majority_vote().
    def _grad_fast_majority_vote(self):
        for k, v in self._coded_grads_buffer.items(): # ~ for each file
            index_pointer = 0
            for j, p in enumerate(self.network.parameters()): # ~ for each layer
                grad_size = reduce((lambda x, y: x * y), p.size())

                for i, elem in enumerate(v): # ~ for each worker with file k, enumeration never used
                    # ~ Pick the gradient for the current layer from the first worker in the group and don't check the rest of them
                    _maj_grad = elem[j]
                    break
                        
                # ~ test
                # lay = 1
                # if k == 0: # ~ k-th group (file)
                # if j == lay: # ~ j-th layer
                    # np.save('BYZSHIELD_draco_lite_aggregation_buffer_file_'+str(k)+'_lay'+str(j), _maj_grad)

                # try:
                    # assert self._grad_aggregate_buffer[j].shape == _maj_grad.shape
                # except AssertionError as e:
                    # warnings.warn("Gradient shapes incompatible, deprecated! ")

                self._draco_lite_aggregation_buffer[k, index_pointer:index_pointer+grad_size] = _maj_grad.reshape(-1)
                index_pointer += grad_size
                

    # ~ reset 1st and 3rd stage output buffers
    def meset_grad_buffer(self):
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape, dtype=float_type)

        self._draco_lite_aggregation_buffer = np.zeros((self._sub_grad_size, self._model_param_counter), dtype=float_type)

    def _LIE_attack_simulation(self):
        """
        Simulating the attack method in: https://arxiv.org/abs/1902.06156
        (Otheriwse, 2-round communication strategy needs to be implemented for each iteration)
        The process is farily simple: (p_mal)_i = mu_i + z^{max} * simga_i, for i in {1, ..., d}
        """
        self.__z = 1.0
        
        # ~ new random choice, this set of Byzantines for ALIE may be different than the Byzantines chosen at the start of iteration, must be WRONG as it returns values in [0,1,...q-1]
        # dummpy_adversarial_nodes = np.random.choice(self.num_workers, self._s, replace=False)
        
        # ~ option 2: choice of Byzantines won't change for ALIE
        dummpy_adversarial_nodes = self._fail_workers[self.cur_step]
        
        # ~ test
        logger.info("DEBUG_PS_BYZ: ALIE byzantines: {}".format(dummpy_adversarial_nodes))
        # logger.info("DEBUG_PS_BYZ: ALIE self._group_list: {}".format(self._group_list))
        
        #################################################################################################################################
        # ~ Code for the fixed disagreement attack (useful with subset assignment)
        # Note: see comments in worker's code as it is almost same
        #################################################################################################################################
        cur_step_adversaries = set(dummpy_adversarial_nodes)
        
        cur_step_honest = self.all_workers - cur_step_adversaries
        
        cur_step_D = set(sorted(cur_step_honest)[:len(cur_step_adversaries)])
        #################################################################################################################################
        
        # ~ first it collects the j-th layer gradient from all files of all workers
        for j, _ in enumerate(self.network.parameters()):
            # aggregate coded gradients from different groups together, this is stupid, try to avoid this
            tempt_grads = []
            for k, v in self._coded_grads_buffer.items(): # ~ k=0...f-1, v: list of workers (ranks) with k
                for i, elem in enumerate(v): # ~ for each worker (rank) in current group v, enumeration is not needed, remove it
                    tempt_grads.append(elem[j]) # ~ collect j-th layer gradient
            mu, sigma = np.mean(tempt_grads, axis=0), np.std(tempt_grads, axis=0) # ~ mean and std for this layer 

            for adv_index in dummpy_adversarial_nodes:
                for k, v in self._coded_grads_buffer.items():
                    if self._err_choice == 'all':
                        if adv_index in self._group_list[k]:
                            _mal_grad = mu + self.__z * sigma # ~ malicious gradient for current layer
                            _relative_index_in_group = self._group_list[k].index(adv_index)
                            assert self._coded_grads_buffer[k][_relative_index_in_group][j].shape == _mal_grad.shape
                            self._coded_grads_buffer[k][_relative_index_in_group][j] =  mu + self.__z * sigma
                    elif self._err_choice == 'fixed_disagreement':
                        # ~ Note: see comments in worker's code as it is almost same
                        
                        group = set(self._group_list[k])
                        
                        cur_group_adversaries = cur_step_adversaries & group
                        
                        cur_group_honest = group - cur_group_adversaries
                        
                        if adv_index in group and cur_group_honest.issubset(cur_step_D):
                            # ~ same as above
                            _mal_grad = mu + self.__z * sigma # ~ malicious gradient for current layer
                            _relative_index_in_group = self._group_list[k].index(adv_index)
                            assert self._coded_grads_buffer[k][_relative_index_in_group][j].shape == _mal_grad.shape
                            self._coded_grads_buffer[k][_relative_index_in_group][j] =  mu + self.__z * sigma                    
                    elif self._err_choice == 'only_majorities':
                    
                        group = set(self._group_list[k])
                        
                        cur_group_adversaries = cur_step_adversaries & group
                        
                        # ~ Distort if the current adversary is in the current group AND there is adversarial majority in the group
                        if adv_index in group and len(cur_group_adversaries) >= (self._group_size + 1)//2:
                        
                            # ~ test
                            # if j == 0: logger.info("DEBUG_PS_BYZ: ALIE worker {} distorts file {} of {} adversaries".format(adv_index, group, len(cur_group_adversaries)))
                        
                            # ~ same as above
                            _mal_grad = mu + self.__z * sigma # ~ malicious gradient for current layer
                            _relative_index_in_group = self._group_list[k].index(adv_index)
                            assert self._coded_grads_buffer[k][_relative_index_in_group][j].shape == _mal_grad.shape
                            self._coded_grads_buffer[k][_relative_index_in_group][j] =  mu + self.__z * sigma 
                        
                        # ~ test
                        # else:
                            # if j == 0: logger.info("DEBUG_PS_BYZ: ALIE worker {} is not in or does not distort file {} of {} adversaries".format(adv_index, group, len(cur_group_adversaries)))
                    else: # ~ invalid
                        assert 0 == 1, "Error: Unknown adversarial distortion choice!"


    # ~ Fall of Empires attack
    def _FoE_attack_simulation(self):
        """
        Simulating the attack method in: https://arxiv.org/abs/1903.03936
        Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation
        """
        # dummpy_adversarial_nodes = np.random.choice(self.num_workers, self._s, replace=False)
        
        # ~ option 2: choice of Byzantines won't change for FoE
        dummpy_adversarial_nodes = self._fail_workers[self.cur_step]
        
        # ~ test
        logger.info("DEBUG_PS_BYZ: FoE byzantines: {}".format(dummpy_adversarial_nodes))
        # logger.info("DEBUG_PS_BYZ: FoE self._group_list: {}".format(self._group_list))
        
        #################################################################################################################################
        # ~ Code for the fixed disagreement attack (useful with subset assignment)
        # Note: see comments in worker's code as it is almost same
        #################################################################################################################################
        cur_step_adversaries = set(dummpy_adversarial_nodes)
        
        cur_step_honest = self.all_workers - cur_step_adversaries
        
        cur_step_D = set(sorted(cur_step_honest)[:len(cur_step_adversaries)])
        #################################################################################################################################        
        
        for j, _ in enumerate(self.network.parameters()):
            tempt_grads = []
            for k, v in self._coded_grads_buffer.items():
                for i, elem in enumerate(v):
                    if self._group_list[k][i] not in dummpy_adversarial_nodes:
                        tempt_grads.append(elem[j])
                        
            # minimum, maximum, mu = np.amin(tempt_grads, axis=0), np.amax(tempt_grads, axis=0), np.mean(tempt_grads, axis=0)
            mu = np.mean(tempt_grads, axis=0)
            
            # ~ test
            # logger.info("DEBUG_PS_BYZ: FoE tempt_grads, mu: {} {}".format(np.shape(tempt_grads), np.shape(mu)))
            
            # OFFSET = 100
            FACTOR = 100
            for adv_index in dummpy_adversarial_nodes:
                for k, v in self._coded_grads_buffer.items():
                    if self._err_choice == 'all':
                        if adv_index in self._group_list[k]:
                            # _mal_grad = v
                            # _mal_grad[mu > 0] = minimum[mu > 0] #- OFFSET*np.ones(np.shape(_mal_grad[mu > 0]))
                            # _mal_grad[mu <= 0] = maximum[mu <= 0] #+ OFFSET*np.ones(np.shape(_mal_grad[mu > 0]))
                            _mal_grad = -FACTOR*mu
                            _relative_index_in_group = self._group_list[k].index(adv_index)
                            assert self._coded_grads_buffer[k][_relative_index_in_group][j].shape == _mal_grad.shape
                            self._coded_grads_buffer[k][_relative_index_in_group][j] = _mal_grad
                    elif self._err_choice == 'fixed_disagreement':
                        # ~ Note: see comments in worker's code as it is almost same
                        group = set(self._group_list[k])
                        
                        cur_group_adversaries = cur_step_adversaries & group
                        
                        cur_group_honest = group - cur_group_adversaries
                        
                        if adv_index in group and cur_group_honest.issubset(cur_step_D):
                            # ~ same as above
                            _mal_grad = -FACTOR*mu
                            _relative_index_in_group = self._group_list[k].index(adv_index)
                            assert self._coded_grads_buffer[k][_relative_index_in_group][j].shape == _mal_grad.shape
                            self._coded_grads_buffer[k][_relative_index_in_group][j] = _mal_grad
                    elif self._err_choice == 'only_majorities':
                    
                        group = set(self._group_list[k])
                        
                        cur_group_adversaries = cur_step_adversaries & group
                        
                        # ~ Distort if the current adversary is in the current group AND there is adversarial majority in the group
                        if adv_index in group and len(cur_group_adversaries) >= (self._group_size + 1)//2:
                        
                            # ~ test
                            # if j == 0: logger.info("DEBUG_PS_BYZ: FoE worker {} distorts file {} of {} adversaries".format(adv_index, group, len(cur_group_adversaries)))
                        
                            # ~ same as above
                            _mal_grad = -FACTOR*mu
                            _relative_index_in_group = self._group_list[k].index(adv_index)
                            assert self._coded_grads_buffer[k][_relative_index_in_group][j].shape == _mal_grad.shape
                            self._coded_grads_buffer[k][_relative_index_in_group][j] = _mal_grad
                        
                        # ~ test
                        # else:
                            # if j == 0: logger.info("DEBUG_PS_BYZ: FoE worker {} is not in or does not distort file {} of {} adversaries".format(adv_index, group, len(cur_group_adversaries)))
                    else: # ~ invalid
                        assert 0 == 1, "Error: Unknown adversarial distortion choice!"
    
    
    # ~ Adversarial detection using clique-finding
    def _attempt_clique_detection(self):
        K = self.num_workers
        r = self._group_size
        
        # ~ Compute the number of joint files of each pair of workers (this is also the number of agreements we need to see in order to draw an edge on the detection graph)
        # For some schemes this may need to be passed as an argument to the master
        if self._approach == "subset":
            # Subset assignment: each pair of workers participates into (K-2) choose (r-2) groups
            pairGroups = ncr(K-2, r-2)
        else:
            # ~ Any other assignment: fetch it from the arguments
            pairGroups = self._pair_groups
            
        # ~ test
        # logger.info("DEBUG_PS_BYZ: K, r, pairGroups: {} {} {}".format(K, r, pairGroups))
        # logger.info("DEBUG_PS_BYZ: Step: {}, self._group_list in _attempt_clique_detection(): {}".format(self.cur_step, self._group_list))

        # Counts the agreements of all worker pairs (x,y), i.e., ranks. WLOG, the records will have x < y since the agreement relation is undirected.
        # collections.Counter() considers non-existent keys as zero, so if two workers (x,y) never agree, that won't be an issue
        agreements = Counter()
    
        for file in self._group_list: # ~ for each file 0...f-1
        
            # ~ workers (ranks) that processed the file
            group = self._group_list[file]
            
            for worker1, worker2 in product(group, group): # ~ for each pair of workers that processed the file
                # ~ Determine whether pair agrees on all layer gradients for the file 
                
                # ~ If worker1, worker2 are the same worker go on. Also, since these agreements are bidirectional we don't want to double-count them, so we consider only worker1 < worker2
                if worker1 == worker2 or worker1 > worker2: continue
                
                # ~ fetch the gradients for all layers for the current file of the current pair {worker1, worker2}
                worker1_grad = self._coded_grads_buffer[file][group.index(worker1)]
                worker2_grad = self._coded_grads_buffer[file][group.index(worker2)]
                
                agreement = True
                
                # ~ for each layer, check exact match
                # torch.nn.Parameter does not support len()
                for layer_idx, _ in enumerate(self.network.parameters()):
                    # if not np.array_equal(worker1_grad[layer_idx], worker2_grad[layer_idx]): # ~ Exact equality check - It does NOT work anymore due to precision issues.
                    if not self.gradientsTolEqual(worker1_grad[layer_idx], worker2_grad[layer_idx]): # ~ tolerance-based equality check
                        agreement = False
                        break
                
                # ~ sorting is used since agreement does not use direction, i.e., agreements[(x,y)] == agreements[(y,x)]     
                if agreement: agreements[tuple([worker1, worker2])] += 1
                
        # ~ test
        # logger.info("DEBUG_PS_BYZ: agreements: {}".format(agreements))
                
        # ~ Detection graph G, encodes the agreements of the workers
        G = nx.Graph()
        G.add_nodes_from(range(1, K+1))
        
        # ~ Draw graph edges based on agreements
        for worker1 in range(1, K+1):
            for worker2 in range(worker1+1, K+1):
                if agreements[(worker1, worker2)] == pairGroups:
                    G.add_edge(worker1, worker2)
        
        # ~ Compute all cliques in G         
        C_vec = list(nx.clique.find_cliques(G))
        
        # ~ test
        # logger.info("DEBUG_PS_BYZ: C_vec: {}".format(C_vec))
        
        # ~ Size of each (if more than one) maximum clique in G
        cliqueNum = float('-inf')
        
        # ~ Identify all maximum cliques in G
        self.maxCliques = set()
        for C in C_vec:
            if len(C) > cliqueNum:
                cliqueNum = len(C)
                self.maxCliques = {tuple(C)}
            elif len(C) == cliqueNum:
                self.maxCliques.add(tuple(C))
                
        # ~ test
        logger.info("DEBUG_PS_BYZ: self.maxCliques: {}".format(self.maxCliques))
        
        # ~ The workers that will be ignored since they have been detected (if any)
        self.detectedWorkers = set()
        if len(self.maxCliques) == 1:
            for C in self.maxCliques: break 
            for worker in range(1, K+1):
                if worker not in C:
                    self.detectedWorkers.add(worker)
                    
            # ~ First, identify a non-distorted file gradient that will act as a placeholder gradient for all fully adversarial groups
            cleanGradient = None
            for file in self._group_list:
                group = self._group_list[file]
                for worker in group:
                    if worker not in self.detectedWorkers:
                        # ~ A non-adversary was found in the current group, store this gradient and break, don't look for another one.
                        cleanGradient = self._coded_grads_buffer[file][group.index(worker)]
                        break
                
                # ~ found a non-distorted gradient, break, don't look for another one.
                if cleanGradient is not None: break
                
            # ~ test
            # match the layer index with subsequent debugging code
            # lay = 0 # ~ layer index (0-indexed)
            # np.save('cleanGradient_layer'+str(lay), cleanGradient[lay])
            
            # ~ This needs to be reset at every iteration
            self.fullyDistortedFiles = []
            
            # ~ Next, for each file (group):
            # If the group is fully honest, don't change anything.
            # If the group is fully adversarial, use the placeholder gradient for all participants. 
            #     But, maybe the correct thing is to ignore this group (implemented after majority voting, see optional if condition in _draco_lite_aggregation() function).
            # If the group is mixed (some adversaries, some honest), substitute all adversarial gradients with an honest one within the group, do not use the placeholder here.
            for file in self._group_list:
                group = self._group_list[file]
                groupSet = set(group)
                if len(groupSet & self.detectedWorkers) == 0:
                    # ~ Fully honest group
                    continue
                elif len(groupSet & self.detectedWorkers) == r:
                    # ~ Fully adversarial group
                    for worker in group:
                        # ~ You must do a deep copy otherwise it will refer to the same memory location
                        self._coded_grads_buffer[file][group.index(worker)] = copy.deepcopy(cleanGradient)
                        
                    # ~ test
                    # logger.info("DEBUG_PS_BYZ: Fully adversarial file: {}".format(file))
                    
                    self.fullyDistortedFiles += [file]
                    
                else:
                    # ~ Mixed group
                    
                    # ~ Identify an honest gradient within the group
                    cleanGroupGradient = None
                    for worker in group:
                        if worker not in self.detectedWorkers:
                            # A non-adversary was found in the current group, store this gradient and break, don't look for another one.
                            cleanGroupGradient = self._coded_grads_buffer[file][group.index(worker)]
                            break
                    
                    # ~ Use the honest gradient instead of all adversarial ones within a group
                    for worker in group:
                        if worker in self.detectedWorkers:
                            # ~ You must do a deep copy otherwise it will refer to the same memory location
                            self._coded_grads_buffer[file][group.index(worker)] = copy.deepcopy(cleanGroupGradient)
            
        # ~ test
        logger.info("DEBUG_PS_BYZ: clique-detectedWorkers: {}".format(self.detectedWorkers))
        
        
    # ~ Adversarial detection using degree-based argument
    # Similar to _attempt_clique_detection(), see comments there.
    def _attempt_degree_detection(self):
        from collections import defaultdict
        K = self.num_workers
        r = self._group_size
        
        # ~ test
        # logger.info("DEBUG_PS_BYZ: K = {}, r = {}".format(K, r))
        
        # ~ New (detection) window.
        # -1 is not important, note that iteration index is 1-indexed.
        if (self.cur_step-1)%self._det_win_length == 0:
            # ~ Counts the agreements of all worker pairs (x,y). WLOG, x < y since the agreement relation is undirected.
            # Note: the count refers to the agreements in all iterations of the current historical window until now.
            # In order to draw an edge between two workers, we expect this number to be: (# of iterations in window so far)*self._pair_groups, i.e.,
            # the pair to agree in ALL files and in ALL iterations. So, for an agreement at an iteration, we will increase it by "self._pair_groups" units.
            # Reset it for each new window.
            self._agreements = Counter()
            
            # ~ For the new detection window, also reset the set of detected workers.
            # Unlike in _attempt_clique_detection(), this is a list. The reason is that if there is a misalignment between the detection window and the adversarial window, 
            # more than q adversaries can be detected. By convention, we will then discard the older ones, so we need to preserve ordering.
            self.detectedWorkers = []
            
        # ~ test
        # logger.info("DEBUG_PS_BYZ: self._agreements BEFORE: {}".format(self._agreements))
            
        # At this point, agreements hashtable "self._agreements" is given as input and we build on it.
        
        for file in self._group_list:
            group = self._group_list[file]
            for worker1, worker2 in product(group, group):
                if worker1 == worker2 or worker1 > worker2: continue
                
                worker1_grad = self._coded_grads_buffer[file][group.index(worker1)]
                worker2_grad = self._coded_grads_buffer[file][group.index(worker2)]
                
                agreement = True
                
                for layer_idx, _ in enumerate(self.network.parameters()):
                    # if not np.array_equal(worker1_grad[layer_idx], worker2_grad[layer_idx]): # ~ Exact equality check - It does NOT work anymore due to precision issues.
                    if not self.gradientsTolEqual(worker1_grad[layer_idx], worker2_grad[layer_idx]): # ~ Tolerance-based equality check
                        agreement = False
                        
                        # if worker1 not in self._fail_workers[self.cur_step] and worker2 not in self._fail_workers[self.cur_step]:
                        # if (worker1 not in self._fail_workers[self.cur_step] and worker2 in self._fail_workers[self.cur_step]) or (worker1 in self._fail_workers[self.cur_step] and worker2 not in self._fail_workers[self.cur_step]):
                        # logger.info("DEBUG_PS_BYZ: Workers {}, {} disagree on file {}: {} with L-2 norm diff = {}, |a-b|2/|b|2 = {}, L-oo norm diff = {}, |a-b|oo/|b|oo = {}, |a| = {}, |b| = {}".format(
                            # worker1, worker2, file, group, 
                            # np.linalg.norm(np.subtract(worker1_grad[layer_idx], worker2_grad[layer_idx])),
                            # np.linalg.norm(np.subtract(worker1_grad[layer_idx], worker2_grad[layer_idx]))/np.linalg.norm(worker2_grad[layer_idx]),
                            # np.linalg.norm(np.subtract(worker1_grad[layer_idx], worker2_grad[layer_idx]).flatten(), ord=np.inf), 
                            # np.linalg.norm(np.subtract(worker1_grad[layer_idx], worker2_grad[layer_idx]).flatten(), ord=np.inf)/np.linalg.norm(worker2_grad[layer_idx].flatten(), ord=np.inf),
                            # np.linalg.norm(worker1_grad[layer_idx]),
                            # np.linalg.norm(worker2_grad[layer_idx])
                            # ))
                            
                            
                        # ~ This is only to catch precision issues, the detection does not rely on it
                        if worker1 not in self._fail_workers[self.cur_step] and worker2 not in self._fail_workers[self.cur_step]:
                            assert 0 == 1, "Error! Honest workers disagree. Check for precision issues."
                        
                        break
                        
                if agreement: self._agreements[tuple([worker1, worker2])] += 1
                
        # ~ test
        # logger.info("DEBUG_PS_BYZ: self._agreements AFTER: {}".format(self._agreements))
        
        # ~ Degree hashtable: key: vertex (worker), value: degree of the vertex in the agreement graph.
        # Note: the degree will be computed based on the agreements accumulated during the current window and not just the current iteration.
        Gdegree = defaultdict(int)
        
        windowIterCtr = (self.cur_step-1)%self._det_win_length + 1
        
        for worker1 in range(1, K+1):
            for worker2 in range(worker1+1, K+1):
                if self._agreements[(worker1, worker2)] == windowIterCtr*self._pair_groups:
                    Gdegree[worker1] += 1
                    Gdegree[worker2] += 1
                    
        # ~ test
        # logger.info("DEBUG_PS_BYZ: Gdegree: {}".format(Gdegree))
        
        # ~ Detect based on minimum allowed degree
        for worker in range(1, K+1):
            if Gdegree[worker] < K - self._s - 1:
                if worker not in self.detectedWorkers: self.detectedWorkers += [worker]
                
                
        logger.info("DEBUG_PS_BYZ: BEFORE degree-detectedWorkers: {}".format(self.detectedWorkers))
        
        # ~ A misalignment between the detection window and the adversarial window can lead to more than q adversaries being detected.
        # Pick the newest q of them.
        self.detectedWorkers = self.detectedWorkers[-self._s:] 
        
        # ~ Find a non-distorted gradient to be placeholder for all fully distorted files, see _attempt_clique_detection()
        cleanGradient = None
        for file in self._group_list:
            group = self._group_list[file]
            for worker in group:
                if worker not in self.detectedWorkers:
                    cleanGradient = self._coded_grads_buffer[file][group.index(worker)]
                    
                    # ~ test
                    # logger.info("DEBUG_PS_BYZ: cleanGradient: group = {}, worker = {}".format(group, worker))
                    
                    break
            
            if cleanGradient is not None: break
            
        self.fullyDistortedFiles = []
        
        # ~ Post-processing of the gradients after detection, see _attempt_clique_detection()
        for file in self._group_list:
            group = self._group_list[file]
            groupSet = set(group)
            if len(groupSet & set(self.detectedWorkers)) == 0:
                # ~ test
                # logger.info("DEBUG_PS_BYZ: un-distorted group = {}".format(group))
                
                continue
            elif len(groupSet & set(self.detectedWorkers)) == r:
                for worker in group:
                    self._coded_grads_buffer[file][group.index(worker)] = copy.deepcopy(cleanGradient)
                    
                # ~ test
                # logger.info("DEBUG_PS_BYZ: fully distorted group = {}".format(group))
                
                self.fullyDistortedFiles += [file]
                
            else:
                # ~ test
                # logger.info("DEBUG_PS_BYZ: partially distorted group = {}".format(group))
                
                cleanGroupGradient = None
                for worker in group:
                    if worker not in self.detectedWorkers:
                        cleanGroupGradient = self._coded_grads_buffer[file][group.index(worker)]
                        
                        # ~ test
                        # logger.info("DEBUG_PS_BYZ: cleanGroupGradient: group = {}, worker = {}".format(group, worker))
                
                        break
                
                for worker in group:
                    if worker in self.detectedWorkers:
                        self._coded_grads_buffer[file][group.index(worker)] = copy.deepcopy(cleanGroupGradient)
        
        # ~ test
        # logger.info("DEBUG_PS_BYZ: self.fullyDistortedFiles: {}".format([self._group_list[file] for file in self.fullyDistortedFiles]))
        logger.info("DEBUG_PS_BYZ: degree-detectedWorkers: {}".format(self.detectedWorkers))


    # ~ Randomly permutes the file assignments by reassigning the values {1,2,...,K} to new values {p(1),p(2),...,p(K)}.
    # Then, for each file (represented by its set of workers - ranks), it re-assigns its worker labels to the permuted ones p(i), i = ... It makes the same change in hashtable from file indices to workers.
    # Hence, the following are updated:
    #   1. self._group_list
    #   2. self.workerFileHt (corresponding to self._group_num at the worker level)
    # Example: If a file is originally [1,2,3], it will be converted to [p(1),p(2),p(3)]. Also, if worker 1 is originally mapped to [a,b,c] then p(1) will be mapped to [a,b,c].
    # Note: This function should run at the PS level and the PS should transmit the updates to the workers.
    # 
    # Parameters:
    # K: total no. of workers
    def permuteListSets(self, K: int):
        # ~ permutedRanks: list where the i-th index will be the random permutation p(i) of the worker with rank i+1.
        # This is the generated bijection between the two sequences. Since this is the list, the order is fixed.
        permutedRanks = list(range(1, K+1))
        random.shuffle(permutedRanks)

        # ~ test
        # print("Workers' permutation: ", [str(rank) + "->" + str(permutedRanks[rank-1]) for rank in range(1, K+1)])
        # logger.info("DEBUG_PS_BYZ: Workers' permutation: {}".format([str(rank) + "->" + str(permutedRanks[rank-1]) for rank in range(1, K+1)]))

        # ~ For each file, replace the worker ranks by their permutations
        for _,file in self._group_list.items():
            for i,worker in enumerate(file):
                # -1 is to offset the 0-indexed list "permutedRanks", see above
                file[i] = permutedRanks[worker-1]
                
        # ~ test  
        # logger.info("DEBUG_PS_BYZ: PERMUTED self._group_list: {}".format(self._group_list))

        # ~ For each worker, map its permutation to the same files since the file assignment has now changed
        workerFileHtNew = {}
        for worker in self.workerFileHt:
            # -1 is to offset the 0-indexed list "permutedRanks", see above
            workerFileHtNew[permutedRanks[worker-1]] = self.workerFileHt[worker]

        # ~ Clear all previous entries and substitute with the new ones
        self.workerFileHt = workerFileHtNew


    # ~ Broadcasts the workers of all files and unicasts the files of each worker. The destinations are workers in both cases.
    # Note: for the unicasts, I use tag = 9 since it is not used anywhere else.
    # 
    # Parameters:
    # K: total no. of workers
    def sendPermToWorkers(self, K: int):
        # ~ Broadcast self._group_list to all workers
        self.comm.bcast(self._group_list, root=0)

        # ~ Unicast files of each worker converting each list to np.ndarray
        for worker in range(1,K+1):
            # Note: the numpy type MUST match the one at the worker level.
            data = np.array(self.workerFileHt[worker]).astype(np.int32)
            self.comm.Send(data, dest=worker, tag=9)
            
    
    # ~ Checks if two given np.ndarray (gradients) a,b are equal allowing for some tolerance.
    def gradientsTolEqual(self, g1, g2):
        # ~ method 1: normalized norm difference and L-oo norm
        # If |b| != 0, check if for L-2 norm: |a-b|/|b| <= tolerance (stronger condition). Also, check if all elements are close in absolute value (L-oo norm) using allclose() (weaker condition),
        # i.e., if for each pair of elements x,y: |x-y| <= tolerance.
        # Note: AND condition is required as DETOX breaks sometimes and decides the minority vote instead of the majority vote.
        # g2Norm = np.linalg.norm(g2)
        # gradDiff = np.subtract(g1, g2)
        # return (g2Norm != 0 and np.linalg.norm(gradDiff)/g2Norm <= self._tolerance) and np.allclose(g1, g2, rtol = self._tolerance, atol = self._tolerance)

        # method 2: normalized norm difference and L-2 norm
        g1Norm = np.linalg.norm(g1)
        g2Norm = np.linalg.norm(g2)
        gradDiff = np.subtract(g1, g2)
        
        # ~ test
        # try:
            # x = (g1Norm == 0 and g2Norm == 0) or (np.linalg.norm(gradDiff)/max(g1Norm, g2Norm) <= self._tolerance)
        # except RuntimeWarning:
            # np.save('gradient_g1', g1)
            # np.save('gradient_g2', g2)
            # logger.info("DEBUG_PS_BYZ: g1Norm: {}, g2Norm: {}".format(g1Norm, g2Norm))
        
        
        # Prevent division by zero
        return (g1Norm == 0 and g2Norm == 0) or (np.linalg.norm(gradDiff)/max(g1Norm, g2Norm) <= self._tolerance)
        
        # method 3: L-oo norm
        # return np.allclose(g1, g2, rtol = 0, atol = self._tolerance)
        