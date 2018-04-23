import numpy as np
import copy
import cv2

from Network import Q_Learner, tf
from Memory import Replay_Buffer
from common import processState, chunks, IMG_H, IMG_W, N_ACTIONS
import gym
import sys


class Agent():
    def __init__(self, env, config, Hidden_Unit_Size=300, Layer_Count=2, Kernel_Size=[8, 4], Stride_Length=[4, 2], Num_Filter=[32, 64], save = False):
        tf.reset_default_graph()
        
        self.env    = env
        self.config = config
        self.save   = save
        self.h_size = Hidden_Unit_Size
        
        self.mainQ      = Q_Learner(IMG_H, IMG_W, self.h_size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter, N_ACTIONS, "MAIN")
        self.targetQ    = Q_Learner(IMG_H, IMG_W, self.h_size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter, N_ACTIONS, "TARGET")

        self.memory  = Replay_Buffer(self.config.Memory_Max_Bytes, 1e3)
        self.session = tf.Session()

        init = tf.global_variables_initializer()
        self.session.run(init)
        print("Created agent with", {'Hidden_Unit_Size': Hidden_Unit_Size, 'Layer_Count': Layer_Count, 'Kernel_Size': Kernel_Size, 'Stride_Length': Stride_Length, 'Num_Filter': Num_Filter})
    def _fill_memory(self):
        for i in range(int(self.config.Pretrain_Episodes)):
            if i != 0 and i % 100 == 0:
                print("Pre-Train Episode:", i)
            reward, ep = self.play_episode(random=True)
            self.memory.add(reward, ep) 

    def play_episode(self, epsilon=0.0, random = True):
        episode = []
        screen = self.env.reset()
        state = processState(screen)
        done = False
        rAll = 0.0
        if not random:
            rnn_state = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))
        while not done:
            if random: #Completely random episode
                action = np.random.randint(0, N_ACTIONS)
            else: #On-Policy Episode
                def get_rnn_state(get):
                    return self.session.run(get, feed_dict={
                        self.mainQ.images      : [state/255.0],
                        self.mainQ.trace_length: 1,
                        self.mainQ.batch_size  : 1,
                        self.mainQ.dropout_p   : 0.75,
                        self.mainQ.rnn_state_in: rnn_state
                    })
                if np.random.rand(1) < epsilon:
                    action = np.random.randint(0, N_ACTIONS)
                    out = get_rnn_state([self.mainQ.rnn_state_out])
                else:
                    action, rnn_state = get([self.mainQ.choice, self.mainQ.rnn_state_out])


            screen_next, reward, done, lives = self.env.step(action)
            state_next = processState(screen_next)
            rAll += reward

            if lives['ale.lives'] != 5:
                episode[-1][0][-1] = True
                break
                
            episode.append(np.reshape(np.array([state.astype(np.uint8), action, reward, state_next.astype(np.uint8), done]), [1, 5]))
            state = state_next
        buffer_array = np.array(episode)
        episode = list(zip(buffer_array))
        return (rAll, episode)  

    def _double_Q_helper(self, isMain, buffer, state_train, batch_size, seq_length):
        if isMain:
            network = self.mainQ
            variable = self.mainQ.choice
        else:
            network = self.targetQ
            variable = self.targetQ.Q

        out = self.session.run(variable, feed_dict={
            network.images      : np.vstack(buffer[:,3]/255.0),
            network.batch_size  : batch_size,
            network.trace_length: seq_length,
            network.rnn_state_in: state_train,
            network.dropout_p   : 1
        })

        return out

    def _train_batch(self, buffer, batch_size, seq_length):
        state_train = (np.zeros([batch_size, self.h_size]), np.zeros([batch_size, self.h_size]))
        #Double DQN
        Q1 = self._double_Q_helper(True, buffer, state_train, batch_size, seq_length)
        Q2 = self._double_Q_helper(False, buffer, state_train, batch_size, seq_length)
        
        end_multiplier = -(buffer[:, 4] - 1)   
        doubleQ = [Q2[a, b, Q1[a,b]] for a in range(0,batch_size) for b in range(0, seq_length)]
        targetQ = doubleQ * end_multiplier
        _, td_error = self.session.run([self.mainQ.train_step, self.mainQ.td_error_sum], feed_dict={
            self.mainQ.batch_size  : batch_size,
            self.mainQ.trace_length: seq_length,
            self.mainQ.ignore_up_to: 0,
            self.mainQ.images      : np.vstack(buffer[:,0]/255.0),
            self.mainQ.target_q    : np.reshape(targetQ,[batch_size, seq_length]),
            self.mainQ.gamma       : 0.99,
            self.mainQ.rewards     : np.reshape(buffer[:,2], [batch_size, seq_length]),
            self.mainQ.actions     : np.reshape(buffer[:,1], [batch_size, seq_length]),
            self.mainQ.dropout_p   : 0.75
        })

        return td_error

    def train(self, episode_count):
        BATCH_SIZE = self.config.Batch_Size
        SEQ_LENGTH = self.config.Sequence_Length
        START_E, END_E = self.config.Noise

        
        e = START_E
        step_drop = (START_E - END_E)/self.config.Annealing_Steps

        trainables = tf.trainable_variables()
        targetOps = self.updateTargetGraph(trainables, self.config.Update_Speed_Tau)

        self._fill_memory()
        rList = []
        for i in range(int(episode_count)):
            #play a trained episode
            rAll, ep = self.play_episode(epsilon=e, random=False)
            if e >= END_E: e -= step_drop
            
            
            #get a sample
            training_batch = self.memory.sample(BATCH_SIZE, SEQ_LENGTH)
            self.memory.add(rAll, ep)

            #train on samples             
            td_error = self._train_batch(training_batch[1], BATCH_SIZE, SEQ_LENGTH)

            #update memory with new td_error
            for idx, err in zip(training_batch[0], td_error):
                self.memory.update(idx, err)
            
            #train on new episode
            #shuffle new episode into chucks of size batch_size*seq_length
            ep = np.reshape(ep, [-1, 5])
            eps = list(chunks(ep, SEQ_LENGTH))
            eps = eps[0:-1]
            np.random.shuffle(eps)
            eps = np.reshape(np.array(eps), [-1, 5])

            eps = list(chunks(eps, SEQ_LENGTH*BATCH_SIZE))
            for x in eps[0:-1]:
                self._train_batch(x, BATCH_SIZE, SEQ_LENGTH)

            #periodically update target network
            if i != 0 and i % self.config.Update_Frequency == 0:
                self.updateTarget(targetOps, self.session)
                print("Updating Target Q-Network", i)
            
            #occasionally display a summary
            rList.append(rAll)
            if i % self.config.Summary_Interval == 0:
                print("Score", (i, np.mean(rList[-self.config.Summary_Interval:]),))
        
    def delete(self):
        self.session.close()
        self.memory = None


    def evaluate(self, num_tests=5):
        return np.random.uniform()

        
    #These functions allows us to update the parameters of our target network with those of the primary network.
    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx,var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        return op_holder

    def updateTarget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)
        total_vars = len(tf.trainable_variables())
        a = tf.trainable_variables()[0].eval(session=sess)
        b = tf.trainable_variables()[total_vars//2].eval(session=sess)
        if a.all() != b.all():
            print("Target Set Failed")