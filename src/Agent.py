import numpy as np
import copy
import cv2
import csv

from Network import Q_Learner, tf
from Memory import Replay_Buffer
from common import processState, chunks, IMG_H, IMG_W, N_ACTIONS
import gym
import sys


class Agent():

    def __init__(self, env, config, Hidden_Unit_Size=300, Layer_Count=2, Kernel_Size=[8, 4], Stride_Length=[4, 2], Num_Filter=[32, 64], save = False, path = None):
        tf.reset_default_graph()
        
        self.env    = env
        self.config = config
        self.save   = save
        self.h_size = Hidden_Unit_Size

        self.mainQ      = Q_Learner(IMG_H, IMG_W, self.h_size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter, N_ACTIONS, "MAIN")
        self.targetQ    = Q_Learner(IMG_H, IMG_W, self.h_size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter, N_ACTIONS, "TARGET")

        self.memory  = Replay_Buffer(int(self.config.Memory_Capacity))
        self.session = tf.Session()

        init = tf.global_variables_initializer()
        self.session.run(init)
        self.desc = {'Hidden_Unit_Size': Hidden_Unit_Size, 'Layer_Count': Layer_Count, 'Kernel_Size': Kernel_Size, 'Stride_Length': Stride_Length, 'Num_Filter': Num_Filter}
        self.ID =  '_'.join([str(a) for a in [Hidden_Unit_Size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter]]).replace('[', '(').replace(']',')')
        print("Created agent with", self.desc)
        self.saver = tf.train.Saver()
        if path != None:
            self.saver.restore(self.session, path)
    
    def _fill_memory(self):
        episodes = int(self.config.Memory_Capacity)
        for i in range(episodes):
            if i != 0 and i % (episodes // 10) == 0:
                print("%"+str(100 * i/episodes))
            reward, ep = self.play_episode(random=True)
            self.memory.add(reward, ep) 

    def play_episode(self, epsilon=0.0, random = True, render = False):
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
                    action, rnn_state = get_rnn_state([self.mainQ.choice, self.mainQ.rnn_state_out])

            if render:
                self.env.render()
            screen_next, reward, done, lives = self.env.step(action)
            state_next = processState(screen_next)
            rAll += reward

            if not render:
                if lives['ale.lives'] != 5 :
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

        log = open('log.csv', 'w')
        log_writer = csv.writer(log, delimiter=',')
        log_writer.writerow(["Episode", "MeanScore", "Epsilon"])


        BATCH_SIZE = self.config.Batch_Size
        SEQ_LENGTH = self.config.Sequence_Length
        START_E, END_E = self.config.Noise

        
        e = START_E
        step_drop = (START_E - END_E)/self.config.Annealing_Steps

        trainables = tf.trainable_variables()
        targetOps = self.updateTargetGraph(trainables, self.config.Update_Speed_Tau)

        print("Pre-Training")

        self._fill_memory()
        self.rList = []
        print("Training for", episode_count, "episodes")
        for i in range(int(episode_count)):
            #play a trained episode
            rAll, ep = self.play_episode(epsilon=e, random=False)
            if e >= END_E: e -= step_drop
            
            
            #get a sample
            training_batch = self.memory.sample(BATCH_SIZE, SEQ_LENGTH)
            self.memory.add(rAll, ep)
            if len(ep) % SEQ_LENGTH == 0:
                epP = np.reshape(ep, [-1, 5])
            else:
                epP = np.reshape(ep[:-(len(ep) % SEQ_LENGTH)], [-1, 5])
            combined_experience = np.concatenate((training_batch[1], epP), 0)
            b_size = combined_experience.shape[0]//SEQ_LENGTH

            


            

            #train on samples             
            td_error = self._train_batch(combined_experience, b_size, SEQ_LENGTH)
            #update memory with new td_error
            for idx, err in zip(training_batch[0], td_error[:BATCH_SIZE]):
                self.memory.update(idx, err)

            #periodically update target network
            if i != 0 and i % self.config.Update_Frequency == 0:
                self.updateTarget(targetOps, self.session)
            
            #occasionally display a summary
            self.rList.append(rAll)
            if i != 0 and i % self.config.Summary_Interval == 0:
                mean = np.mean(self.rList[-self.config.Summary_Interval:])
                print("Score", (i, mean, e))
                log_writer.writerow([i, mean, e])
                log.flush()

            if self.save and i % self.config.Save_Interval == 0:
                save_path = self.saver.save(self.session, "./models/" + self.ID + '.ckpt')
                print("Model saved in path %s" % save_path)
        
    def delete(self):
        self.session.close()
        self.memory = None


    def evaluate(self):
        return np.mean(self.rList)

        
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