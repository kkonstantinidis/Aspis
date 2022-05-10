from .utils import * # ~ will import gradient compression compress(), err_simulation(), accuracy(), models from "model_ops", ...
from .baseline_worker import DistributedWorker

import logging # ~ for comments on logging see "distributed_nn.py"

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ByzshieldWorker(DistributedWorker):
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.rank = comm.Get_rank() # rank of this Worker
        #self.status = MPI.Status()
        self.cur_step = 0 # ~ current step index (across all epochs), updated globally at the master
        self.next_step = 0 # we will fetch this one from parameter server

        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.lr = kwargs['learning_rate']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._adversery = kwargs['adversery'] # ~ never used
        self._err_mode = kwargs['err_mode']
        self._group_list = kwargs['group_list'] # ~ dictionary from file 0...f-1 to list of workers (ranks) that have it
        self._train_dir = kwargs['train_dir']
        self._checkpoint_step = kwargs['checkpoint_step']
        self._eval_freq = kwargs['eval_freq']
        self._max_steps = kwargs['max_steps']
        self._lis_simulation = kwargs['lis_simulation']

        self._fail_workers = kwargs['adversaries']

        self._group_seeds = kwargs['group_seeds'] # ~ list of distinct files 0...f-1
        self._group_num = kwargs['group_num'] # ~ list of files for caller worker (rank)
        self._group_size = len(self._group_list[0]) # ~ == r
        self._compress_grad = kwargs['compress_grad']
        self._device = kwargs['device']
        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = [] # ~ never used
        
        self.ell = len(self._group_num) # ~ computation load per worker ("l" in paper)
        self.F = len(self._group_seeds) # ~ total number of files
        self.seeds = kwargs['seeds'] # ~ random seeds to be used by all workers for each epoch
        
        self._update_mode = kwargs['update_mode']
        
        self._err_choice = kwargs['err_choice'] # ~ if set to 'fixed_disagreement', the adversaries will only distort files of groups where all honest workers are in a fixed disagreement set. If set to 'all', the adversaries will distort all files.
        
        # ~ Ranks of all workers
        self.all_workers = set(range(1, self.world_size))
        
        self._permute_files = kwargs['permute_files']
        
        # ~ test
        # if self.rank == 1:
            # logger.info("DEBUG_W_BYZ: self._group_list: {} {}".format(self._group_list, type(self._group_list)))
            # logger.info("DEBUG_W_BYZ: self._group_num: {} {}".format(self._group_num, type(self._group_num)))
            # logger.info("DEBUG_W_BYZ: self._group_seeds: {} {}".format(self._group_seeds, type(self._group_seeds)))
            # logger.info("DEBUG_W_BYZ: self.ell: {}".format(self.ell))
            # logger.info("DEBUG_W_BYZ: self._err_choice: {}".format(self._err_choice))
            # logger.info("DEBUG_W_BYZ: float_type: {}".format(float_type))
            # logger.info("DEBUG_W_BYZ: self._permute_files: {}".format(self._permute_files))

    def build_model(self):
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

        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf() # ~ creates a buffer equal to the model size & shape
        
        # ~ test
        if self.rank == 1:
            logger.info("DEBUG_W_BYZ: Torch device: {} {}".format(self._device, type(self._device)))
            
            # ~ Total model size in bytes
            # logger.info("DEBUG_W_BYZ: self.network bytes: {}".format(sum([param.nelement()*param.element_size() for param in self.network.parameters()])))

        self.network.to(self._device)


    def train(self, train_loader, test_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1

        self.sync_fetch_step() # ~ updates self.next_step to 1
        # do some sync check here
        assert(self.update_step()) # ~ verifies that self.cur_step (== 0) is not equal to self.next_step (== 1) and sets self.cur_step = 1
        # assert(self.cur_step == STEP_START_) # ~ verifies above change
        if self._checkpoint_step == 0:
            assert(self.cur_step == STEP_START_)
        else:
            assert(self.cur_step == int(self._checkpoint_step)+1)

        # number of batches in one epoch
        # num_batch_per_epoch = len(train_loader.dataset) / self.batch_size # ~ never used
        batch_idx = -1
        epoch_idx = 0
        epoch_avg_loss = 0
        iteration_last_step = 0
        iter_start_time = 0
        first = True
        iter_avg_prec1 = 0 # ~ never used
        iter_avg_prec5 = 0 # ~ never used
        # use following flags to achieve letting each worker compute more batches
        # should_enter_next = False # ~ never used

        logger.info("W_BYZ: Worker {}: starting training".format(self.rank))
        # start the training process
        for num_epoch in range(self.max_epochs):
            # after each epoch we need to make sure workers in the same group re-shuffling using the same seed
            # ~ all workers will use the same seed but it needs to change for each epoch, 
            # you may change this to provide a random seed to torch but it should be the same for everyone, so need to set np.random.seed()
            # torch.manual_seed(42+num_epoch)
            
            # ~ test
            # torch.manual_seed(428)
            
            torch.manual_seed(self.seeds[num_epoch]+num_epoch)
            
            # ~test
            # logger.info("DEBUG_W_BYZ: torch.manual_seed: {}".format(self.seeds[num_epoch]+num_epoch))
            
            for batch_idx, (train_image_batch, train_label_batch) in enumerate(train_loader):
                # worker exit task
                if self.cur_step == self._max_steps:
                    break
                X_batch, y_batch = train_image_batch, train_label_batch # ~ tensors
                
                # ~ test
                # logger.info("DEBUG_W_BYZ: y_batch: {}".format(y_batch))
                # if self.rank == 1:
                    # logger.info("DEBUG_W_BYZ: X_batch: {}, {}, {}, {}, {}, {}".format(type(X_batch), X_batch.dtype, X_batch.shape, X_batch.nelement(), X_batch.element_size(), X_batch.element_size()*X_batch.nelement()))
                    # logger.info("DEBUG_W_BYZ: y_batch: {}, {}, {}, {}, {}, {}".format(type(y_batch), y_batch.dtype, y_batch.shape, y_batch.nelement(), y_batch.element_size(), y_batch.element_size()*y_batch.nelement()))
                    # logger.info("DEBUG_W_BYZ: train_image_batch: {}, {}".format(type(train_image_batch), train_image_batch.shape))
                    # logger.info("DEBUG_W_BYZ: train_label_batch: {}, {}".format(type(train_label_batch), train_label_batch.shape))
                
                # ~ initialize empty np.ndarray to collect the gradients for all files of current worker, 
                # this is to avoid extra communication
                grads = []
                # ctr = 0 # ~ test
                for p in self.network.parameters():
                    # method 1: initialize to an empty array (0-th dimension) with same remaining dimensions [1,...], needs to be paired with method 1 later on ...
                    # grads.append(np.empty((0,)+p.shape[1:]).astype(float_type)) # torch.Size is in fact a tuple, so it supports the same operations
                    
                    # method 2: pre-allocate an array for all files of the worker (0-th dimension) with same remaining dimensions [1,...], needs to be paired with method 2 later on ...
                    grads.append(np.empty((self.ell*p.shape[0],)+p.shape[1:], dtype=float_type)) # torch.Size is in fact a tuple, so it supports the same operations
                    
                    # ~ test
                    # just prints the current layer's dimensions 1,2,... with the 0-th dimension zeroed
                    # if self.rank == 1:
                        # logger.info("DEBUG_W_BYZ: p.shape[0]: {}".format(p.shape))
                        # logger.info("DEBUG_W_BYZ: grads[i].shape: {}".format(grads[ctr].shape))
                    # ctr += 1
                
                # ~ test
                # if self.rank == 1: logger.info("DEBUG_W_BYZ: grads in bytes: {}".format(sum([g.nbytes for g in grads])))
                    
                    
                while True:
                    # the worker shouldn't know the current global step except received the message from parameter server
                    self.async_fetch_step()
                    # the only way every worker know which step they're currently on is to check the cur step variable
                    updated = self.update_step() # ~ updates ctr of step +1 or something from the PS
                    if (not updated) and (not first):
                        # wait here unitl enter next step
                        continue
                    # the real start point of this iteration
                    iter_start_time = time.time()
                    first = False
                    # should_enter_next = False # ~ never used
                    logger.info("W_BYZ: Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))
                    
                    # ~ All adversaries (ranks) of current iteration, in general this set will be fixed
                    cur_step_adversaries = set(self._fail_workers[self.cur_step])
                    
                    # ~ All non-adversaries (ranks) of current iteration, in general this set will be fixed
                    cur_step_honest = self.all_workers - cur_step_adversaries
                    
                    # ~ Non-adversaries (ranks) of current iteration with whom adversaries will disagree, in general this set will be fixed
                    # Recall that the q adversaries afford to disagree with at most q non-adversaries to evade detection
                    # Sorting is needed to guarantee consistency among all adversaries (but may not be needed in Python)
                    cur_step_D = set(sorted(cur_step_honest)[:len(cur_step_adversaries)])
                    
                    # ~ test
                    if self.rank == 1:
                        logger.info("W_BYZ: Step {} adversaries, Honest, Disagreement set: {} {} {}".format(self.cur_step, cur_step_adversaries, cur_step_honest, cur_step_D))
                            
                    fetch_weight_start_time = time.time()
                    self.async_fetch_weights_bcast()
                    fetch_weight_duration = time.time() - fetch_weight_start_time
                    
                    # ~ test
                    # Save model at the worker level after it is received from the PS
                    # for lay,x in enumerate(self.network.parameters()):
                        # np.save('worker'+str(self.rank)+'_model_layer'+str(lay), x.detach().numpy())
                    
                    # ~ test
                    # for file in self._group_num:
                        # if self.rank == 2:
                            # low_ind = (self.batch_size//self.F)*file
                            # high_ind = (self.batch_size//self.F)*(file+1)
                            # logger.info("DEBUG_W_BYZ: worker's file: {}".format(file))
                            # logger.info("DEBUG_W_BYZ: worker's file sample indices (slice): {}".format(range(low_ind, high_ind)))
                            # logger.info("DEBUG_W_BYZ: worker's file X_batch's tensor dimensions: {}".format(X_batch[low_ind:high_ind,...].shape))
                            # logger.info("DEBUG_W_BYZ: worker's file y_batch's tensor dimensions: {}".format(y_batch[low_ind:high_ind].shape))
                            # logger.info("DEBUG_W_BYZ: worker's file X_batch's tensor: {}".format(X_batch[low_ind:high_ind,...]))
                            # logger.info("DEBUG_W_BYZ: worker's file y_batch's tensor: {}".format(y_batch[low_ind:high_ind]))
                    
                    # Initialize variables to hold computation/communication time for all files of the worker
                    forward_duration = 0
                    backward_duration = 0
                    # save_duration = 0 # ~ test
                    for fileInd, file in enumerate(self._group_num): # ~ for each ByzShield file do the forward-backward step
                    
                        # ~ test
                        # if file == 0: # ~ for one file
                        # np.save('BYZSHIELD_file_'+str(file)+'_worker_'+str(self.rank)+'_X_batch', 
                            # X_batch[(self.batch_size//self.F)*file:(self.batch_size//self.F)*(file+1),...].detach().numpy())
                        # np.save('BYZSHIELD_file_'+str(file)+'_worker_'+str(self.rank)+'_y_batch', 
                            # y_batch[(self.batch_size//self.F)*file:(self.batch_size//self.F)*(file+1)].detach().numpy())
                        
                        # ~ Move NN to torch device (CPU or GPU)
                        self.network.to(self._device)
                        
                        self.network.train()
                        self.optimizer.zero_grad()
                        # forward step
                        forward_start_time = time.time()
                        
                        # ~ Pick only the samples of current file from batch, and preserve the rest of the dimensions of the tensor.
                        # Also, move input data and labels to torch device (CPU or GPU).
                        logits = self.network(X_batch[(self.batch_size//self.F)*file:(self.batch_size//self.F)*(file+1),...].to(self._device))
                        loss = self.criterion(logits, y_batch[(self.batch_size//self.F)*file:(self.batch_size//self.F)*(file+1)].to(self._device))
                        
                        forward_duration += time.time()-forward_start_time
                        
                        # ~ test
                        # np.save('BYZSHIELD_file_'+str(file)+'_worker_'+str(self.rank)+'_logits', logits.detach().numpy())
                        
                        # ~ test
                        # if self.rank == 1:
                            # logger.info("DEBUG_W_BYZ: logits: {} {} {} {}".format(type(logits), logits.dtype, logits.shape, logits.element_size()*logits.nelement()))
                            
                        # ~ test
                        # if file == 0: # ~ for one file
                        # np.save('BYZSHIELD_worker_'+str(self.rank)+'_file_'+str(file)+'_logits_step'+str(self.cur_step), logits.detach().numpy())

                        # ~ test
                        # if self.rank == 2:
                            # logger.info("DEBUG_W_BYZ: forward_duration[i]: {}".format(time.time()-forward_start_time))
                                
                        # backward step
                        backward_start_time = time.time()
                        loss.backward()
                        backward_duration += time.time() - backward_start_time
                        
                        # ~ test
                        # if self.rank == 2:
                            # logger.info("DEBUG_W_BYZ: backward_duration[i]: {}".format(time.time()-backward_start_time))

                        # ~ (list) the shape of these gradients depends on the network layers only, not on batchsize or other parameters, the length of the list is equal to the number of layers, see DEBUG below
                        if self._device == torch.device("cpu"):
                            file_grads = [p.grad.detach().numpy().astype(float_type) for p in self.network.parameters()]
                        else:
                            # ~ If CUDA is used, we need to copy the tensor to the main memory first
                            file_grads = [p.grad.detach().cpu().numpy().astype(float_type) for p in self.network.parameters()]
                        
                        # ~ ASPIS: If caller rank is adversarial, distort the current file if its group is fully adversarial or all its non-adversaries are in the disagreements set
                        # Disabled if ALIE or FoE is enabled.
                        if self.rank in self._fail_workers[self.cur_step] and self._err_choice == 'fixed_disagreement' and self._lis_simulation != "simulate" and self._err_mode != "foe":
                            
                            # ~ test
                            # logger.info("DEBUG_W_BYZ: FIXED DISAGREEMENT ATTACK")
                            
                            # ~ Group of workers (ranks) processing the current file
                            group = set(self._group_list[file])
                            
                            # ~ Set of adversaries in current group
                            cur_group_adversaries = cur_step_adversaries & group
                            
                            # ~ Set of non-adversaries in current group
                            cur_group_honest = group - cur_group_adversaries
                            
                            # ~ test
                            # if self.rank == 1:
                                # logger.info("DEBUG_W_BYZ: self._fail_workers[self.cur_step]: {} {}".format(self._fail_workers[self.cur_step], type(self._fail_workers[self.cur_step])))
                                # logger.info("DEBUG_W_BYZ: group: {} {}".format(group, type(group)))
                                # logger.info("DEBUG_W_BYZ: group: {} {} {}".format(group, cur_group_adversaries, cur_group_honest))
                                
                            
                            # ~ An adversary will decide to distort or not current file
                            if cur_group_honest.issubset(cur_step_D):
                                # ~ Either the group is fully adversarial (cur_group_honest == set()) or its non-adversaries are in the "disagreement set". The adversary chooses to distort.
                                
                                # ~ Distort all layers of the file
                                for i in range(len(file_grads)):
                                    # ~ The actual type of distortion will depend on "self._err_mode"
                                    file_grads[i] = err_simulation(file_grads[i], self._err_mode)
                                    
                        # ~ ASPIS+: If caller rank is adversarial, distort the current file if its group has adversarial majority
                        # Disabled if ALIE or FoE is enabled.
                        if self.rank in self._fail_workers[self.cur_step] and self._err_choice == 'only_majorities' and self._lis_simulation != "simulate" and self._err_mode != "foe":
                            group = set(self._group_list[file])
                            cur_group_adversaries = cur_step_adversaries & group
                            cur_group_honest = group - cur_group_adversaries
                            
                            # ~ test
                            # logger.info("DEBUG_W_BYZ: self._fail_workers[self.cur_step]: {}".format(self._fail_workers[self.cur_step]))
                            # logger.info("DEBUG_W_BYZ: My rank: {}, group: {} {} {}".format(self.rank, group, cur_group_adversaries, cur_group_honest))
                                
                                
                            if len(cur_group_adversaries) > len(cur_group_honest):
                                # ~ test
                                # logger.info("DEBUG_W_BYZ: Adversary {} distorts file {}. The adversarial set is: {}".format(self.rank, group, cur_step_adversaries))
                            
                                for i in range(len(file_grads)):
                                    file_grads[i] = err_simulation(file_grads[i], self._err_mode)
                            
                        # ~ test
                        # if self.rank == 1: logger.info("DEBUG_W_BYZ: file_grads in bytes: {}".format(sum([g.nbytes for g in file_grads])))

                        # save_start_time = time.time() # ~ test
                        
                        # method 1: needs to be paired with method 1 above ...
                        # for i in range(len(grads)):
                            # grads[i] = np.append(grads[i], file_grads[i], axis = 0) # ~ np.append() is not in-place
                        
                        # method 2 (fast due to pre-allocation): needs to be paired with method 2 above ...
                        for i, p in enumerate(self.network.parameters()):
                            grads[i][p.shape[0]*fileInd:p.shape[0]*(fileInd+1),...] = file_grads[i]
                            
                        # save_duration += time.time()-save_start_time
                        
                        # ~ test
                        # if self.rank == 1:
                            # logger.info("DEBUG_W_BYZ: length of file_grads of file {}: {}".format(file, len(file_grads)))
                            # for i in range(len(file_grads)):
                                # logger.info("DEBUG_W_BYZ: file_grads[i] of file {}: {}, {}".format(file, type(file_grads[i]), file_grads[i].shape))
                                
                            # logger.info("DEBUG_W_BYZ: length of grads of file {}: {}".format(file, len(grads)))
                            # for i in range(len(grads)): # ~ for each iteration over "file" (outer loop), grads[i] should increase its 0-th dimension by file_grads[i].shape[0] (if method 1 is used)
                                # logger.info("DEBUG_W_BYZ: grads[i]: {}, {}".format(type(grads[i]), grads[i].shape))
                                
                            # ~ save gradient of current file for layer lay
                            # for lay in range(len(file_grads)):
                                # np.save('worker'+str(self.rank)+'_grads_layer'+str(lay)+'_file'+str(file), file_grads[lay])
                                
                        # ~ test
                        # lay = 0
                        # if self.cur_step == 7:
                            # np.save('worker'+str(self.rank)+'_grads_layer'+str(lay)+'_file'+str(file), file_grads[lay])

                        # ~ we still pick only the appropriate batch's labels & torch.long() converts to torch.int64
                        prec1, prec5 = accuracy(logits.detach(), train_label_batch[(self.batch_size//self.F)*file:(self.batch_size//self.F)*(file+1)].long().to(self._device), topk=(1, 5))
                        
                    # in current setting each group cotains k workers, we let each worker calculate k same batches
                    c_start = time.time()
                    self._send_grads(grads)
                    c_duration = time.time() - c_start

                    computation_time = forward_duration + backward_duration
                    logger.info('W_BYZ: byzshield_worker: {}, Step: {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Time Cost: {:.4f}, Comp: {:.4f}, Comm: {:.4f} + {:.4f}, Prec@1: {}, Prec@5: {}'.format(self.rank,
                         self.cur_step, num_epoch, batch_idx * self.batch_size, len(train_loader.dataset), 
                            (100. * (batch_idx * self.batch_size) / len(train_loader.dataset)), loss.item(), time.time()-iter_start_time, computation_time, fetch_weight_duration, c_duration, prec1.cpu().numpy()[0], prec5.cpu().numpy()[0]))
                    
                    # ~ test
                    # logger.info('W_BYZ: byzshield_worker: {}, Step: {}, Epoch: {}, Save Time Cost: {:.4f}'.format(self.rank, self.cur_step, num_epoch, save_duration))
                    
                    if self.cur_step%self._eval_freq == 0 and self.rank==1:
                        #self._save_model(file_path=self._generate_model_path())
                        self._save_model(file_path=self._generate_model_path())
                        
                    # ~ Worker receives the permuted file assignments
                    if self._permute_files == 'yes':
                        # logger.info("DEBUG_W_BYZ: (BEFORE) Worker rank {}: {} {} {} {} {} {}".format(self.rank, self._group_list, type(self._group_list), self._group_num, type(self._group_num), self._group_seeds, type(self._group_seeds)))
                        
                        self.recvPermFromPS()
                        
                        # logger.info("DEBUG_W_BYZ: (AFTER) Worker rank {}: {} {} {} {} {} {}".format(self.rank, self._group_list, type(self._group_list), self._group_num, type(self._group_num), self._group_seeds, type(self._group_seeds)))
                        
                    break # ~ breaks here for everyone after one step of training (current batch)

    def _send_grads(self, grads):
        # ~ MPI tag for these transmissions is 88+(layer index), e.g., for Lenet 88,89,...,95
        # ~ test
        # if self.rank == 1:
            # logger.info("DEBUG_W_BYZ: length of grads: {}".format(len(grads)))
            # for i, grad in enumerate(grads):
                # _compressed_grad = compress(grad)  
                # logger.info("DEBUG_W_BYZ: _compressed_grad: {} {} {}".format(type(_compressed_grad), len(_compressed_grad), getsizeof(_compressed_grad)))
        
        req_send_check = []
        #for i, grad in enumerate(reversed(grads)):
        for i, grad in enumerate(grads): # ~ for each layer
            # ~ test
            # if self.rank == 1:
                # # logger.info("DEBUG_W_BYZ: tag=88+i: {}".format(88+i))
                # if i == 0: # ~ for first layer
                    # logger.info("DEBUG_W_BYZ: Worker {} sending file gradients for layer {}: {}, {}".format(self.rank, i, type(grad), grad.shape))
                
            if len(req_send_check) != 0: # ~ wait on previous request before sending the current one
                req_send_check[-1].wait()
            if self._lis_simulation == "simulate": # ~ ALIE attack will be "simulated" at the PS, all workers send honest gradients
                if self._compress_grad=='compress':
                    if self._update_mode=="sign-sgd":
                        # signSGD worker side
                        grad = np.sign(grad).astype(np.int8)
                    _compressed_grad = compress(grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
            else:
                # ~ ASPIS: adversarial workers have already distorted some files in case of the 'fixed_disagreement' attack
                if self.rank in self._fail_workers[self.cur_step] and self._err_choice == 'all':
                    simulation_grad = err_simulation(grad, self._err_mode)
                    if self._update_mode=="sign-sgd":
                        # signSGD worker side
                        simulation_grad = np.sign(simulation_grad).astype(np.int8)
                    if self._compress_grad=='compress':
                        _compressed_grad = compress(simulation_grad)
                        req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                    else:
                        req_isend = self.comm.Isend([simulation_grad, MPI.DOUBLE], dest=0, tag=88+i)
                    req_send_check.append(req_isend)
                else:
                    if self._compress_grad=='compress':
                        if self._update_mode=="sign-sgd":
                            # signSGD worker side
                            grad = np.sign(grad).astype(np.int8)
                        _compressed_grad = compress(grad)
                        req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                    else:
                        req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                    req_send_check.append(req_isend)
        req_send_check[-1].wait()


    '''
    def _send_grads(self, grads):
        req_send_check = []
        #for i, grad in enumerate(reversed(grads)):
        for i, grad in enumerate(grads):
            if len(req_send_check) != 0:
                req_send_check[-1].wait()
            if self.rank in self._fail_workers[self.cur_step]:
                simulation_grad = err_simulation(grad, self._err_mode)
                if self._compress_grad=='compress':
                    _compressed_grad = compress(simulation_grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([simulation_grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
            else:
                if self._compress_grad=='compress':
                    _compressed_grad = compress(grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
        req_send_check[-1].wait()
    '''
    
    
    # ~ Receive the new file permutation from the PS
    def recvPermFromPS(self):
        # ~ Receive self._group_list from the PS
        self._group_list = self.comm.bcast(self._group_list, root=0)

        # ~ Receive self._group_num from the PS. Use a np.ndarray as a buffer and allocate as much space as the no. of files per worker.
        # Note: the numpy type MUST match the one at the PS level.
        group_num_np = np.empty((self.ell,), dtype=np.int32)
        self.comm.Recv(group_num_np, source=0, tag=9)
        self._group_num = group_num_np.tolist()
        
        # ~ test  
        # logger.info("DEBUG_PS_BYZ: PERMUTED worker's {} self._group_list: {}".format(self.rank, self._group_list))
        # logger.info("DEBUG_PS_BYZ: PERMUTED worker's {} self._group_num: {}".format(self.rank, self._group_num))