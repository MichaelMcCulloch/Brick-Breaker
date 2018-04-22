import numpy as np
import cv2

from Network import Q_Learner, tf
from Memory import Replay_Buffer
import gym


class Agent():
    def __init__(self, env, config, Hidden_Unit_Size=300, Layer_Count=2, Kernel_Size=[8, 4], Stride_Length=[4, 2], Num_Filter=[32, 64], save = False):
        tf.reset_default_graph()
        self.img_h, self.img_w  = 84, 84
        self.env = env
        self.config = config
        self.save = save
        self.h_size = Hidden_Unit_Size
        
        self.mainQ      = Q_Learner(self.img_h, self.img_w, self.h_size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter, 4, "MAIN")
        self.targetQ    = Q_Learner(self.img_h, self.img_w, self.h_size, Layer_Count, Kernel_Size, Stride_Length, Num_Filter, 4, "TARGET")

        self.memory = Replay_Buffer(self.config.Memory_Max_Bytes, 10e2)

        self.session = tf.Session()

    def train(self, episode_count):
        BATCH_SIZE = self.config.Batch_Size
        SEQ_LENGTH = self.config.Sequence_Length
        START_E, END_E = self.config.Noise
        ANNEALING_STEPS = self.config.Annealing_Steps

        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()
        targetOps = self.updateTargetGraph(trainables, self.config.Update_Speed_Tau)
        e = START_E
        step_drop = (START_E - END_E)/ANNEALING_STEPS

        jList = []
        rList = []
        total_steps = 0
        
        self.session.run(init)
        #set the target to be equal to the primary
        self.updateTarget(targetOps, self.session)
        for i in range(int(episode_count)):
            episode_buffer = []
            screen = self.env.reset()
            state = self.processState(screen)
            done = False
            rAll = 0
            j = 0

            rnn_state = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))
            while not done:
                j += 1
                #Do a random step
                if np.random.rand(1) < e or total_steps < config.Pretrain_Episodes:
                    rnn_state = self.session.run(self.mainQ.rnn_state_out, feed_dict= {
                        self.mainQ.images : [state/255.0],
                        self.mainQ.trace_length : 1,
                        self.mainQ.batch_size : 1,
                        self.mainQ.dropout_p : 0.75,
                        self.mainQ.rnn_state_in :  rnn_state
                    })
                    action = np.random.randint(0, 4)
                #Do a trained step
                else:
                    action, rnn_state = self.session.run([self.mainQ.choice, self.mainQ.rnn_state_out], feed_dict={
                        self.mainQ.images : [state/255.0],
                        self.mainQ.trace_length : 1,
                        self.mainQ.batch_size : 1,
                        self.mainQ.dropout_p : 0.75,
                        self.mainQ.rnn_state_in :  rnn_state
                    })
                    action = action[0]
                screen_next, reward, done, lives = self.env.step(action)
                if lives['ale.lives'] != 5: done = True
                
                state_next = self.processState(screen_next)
                total_steps += 1

                episode_buffer.append(np.reshape(np.array([state, action, reward, done]), [1,4]))
                if total_steps >= self.config.Pretrain_Episodes:
                    if e >= self.config.Noise[1]: e -= step_drop
                    if total_steps % self.config.Update_Frequency == 0:
                        self.updateTarget(targetOps, self.session)
                        state_train = (np.zeros([BATCH_SIZE, self.h_size]), np.zeros([BATCH_SIZE, self.h_size]))
                        trainBatch = self.memory.sample(BATCH_SIZE, SEQ_LENGTH)

                        #Double DQN Update:
                        target_Q = self.session.run(self.targetQ.max_Q, feed_dict={
                            self.targetQ.batch_size : BATCH_SIZE,
                            self.targetQ.trace_length : SEQ_LENGTH,
                            self.targetQ.images: np.vstack(trainBatch[:,0]/255.0),
                            self.targetQ.dropout_p : 1
                        })

                        _, loss_q, qs = self.session.run([self.mainQ.train_step, self.mainQ.q_loss, self.mainQ.Q], feed_dict={
                            self.mainQ.batch_size : BATCH_SIZE,
                            self.mainQ.trace_length : SEQ_LENGTH,
                            self.mainQ.ignore_up_to : 4,
                            self.mainQ.images : np.vstack(trainBatch[:,0]/255.0),
                            self.mainQ.target_q : target_Q,
                            self.mainQ.gamma : 0.99,
                            self.mainQ.rewards : trainBatch[:,2],
                            self.mainQ.actions : trainBatch[:,1],
                            self.mainQ.dropout_p : 0.75
                        })



    def evaluate(self, num_tests=5):
        return np.random.uniform()

    def processState(self, state1):
        greyscale = np.asarray(state1).sum(axis=2)/3
        cropped = greyscale[17:, :]
        scaled = cv2.resize(cropped, dsize=(self.img_h, self.img_w), interpolation=cv2.INTER_CUBIC)
        
        return np.reshape(scaled,[self.img_h * self.img_w])
        
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