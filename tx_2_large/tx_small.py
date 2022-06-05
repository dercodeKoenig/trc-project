import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import ta
from tqdm import tqdm
import random
import pickle
from collections import deque
import math
import time


name = "dqn_trading_transformer"
log_folder = "./"
candles_dir = "../candles/"

training_parallel = 16
warmup_parallel = 32
warmup_steps = 5000

batch_size = 64
gamma = 0.99
memory_size = 3000000
lr  = 0.0005
seq_len = 550

soft_reward_inc = 1.05
comission = 10/100000

resume = True
#resume = False

def sample_to_x(sample):
        
                current_close = sample[-1].c

                prev_close = [candle.c for candle in sample]
                prev_high = [candle.h for candle in sample]
                prev_low = [candle.l for candle in sample]
                

                prev_sma21 = [candle.sma21 for candle in sample]
                prev_sma200 = [candle.sma200 for candle in sample]
                
                
                prev_sma21_relative = [(prev_close[o] - prev_sma21[o]) / prev_sma21[o]*100 for o in range(seq_len)]
                prev_sma200_relative = [(prev_close[o] - prev_sma200[o]) / prev_sma200[o]*100 for o in range(seq_len)]

                prev_close_relative = [0] + [(prev_close[o+1] - prev_close[o]) / prev_close[o]*1000 for o in range(seq_len-1)]
                prev_high_relative = [(prev_close[o] - prev_high[o]) / prev_close[o]*1000 for o in range(seq_len)]
                prev_low_relative = [(prev_close[o] - prev_low[o]) / prev_close[o]*1000 for o in range(seq_len)]
                

                
                prev_rsi_14 = [candle.rsi14 for candle in sample]
                

                x = []
                for o in range(len(prev_close)):
                    ts = []

                    
                    ts.append(prev_close_relative[o])
                    ts.append(prev_high_relative[o])
                    ts.append(prev_low_relative[o])
                    
                    ts.append(prev_sma21_relative[o])
                    ts.append(prev_sma200_relative[o])
                    
                    ts.append(prev_rsi_14[o])

                    x.append(ts)

                x = np.array(x)
                return x
        


def Load(file):
    f = open(file, "rb")
    obj = pickle.load(f)
    f.close()
    return obj





log_interval = 4*24 # environment logs daily returns

class candle_class:
    pass
  
order_value = 1000


class environment():

  def __init__(self):
    pass


  def _next_observation(self):
            candles = self.candles[self.current_step - seq_len + 1:self.current_step + 1]
            
            inference_data = sample_to_x(candles)
            
            return inference_data, np.array([self.position, math.tanh(self.current_win)])

  
  def reset(self, first_reset = False):
    self.candles = None
    candles_files = os.listdir(candles_dir)
    use_file = candles_dir+random.choice(candles_files)
    #print(use_file)
    self.candles = Load(use_file)
    
    
    
    self.current_step = 200+seq_len if first_reset == False else random.randint(200+seq_len,len(self.candles) - 1000)
    self.position = 0
    self.entry_price = 0
    self.win = 0
    self.current_win = 0
    self.startindex = self.current_step
    self.last_reward = 0
    self.reward_tr_given = 0
    self.reward_since_last_log = 0
    self.closed_trades_since_last_log = 0

    return self._next_observation()

  
  def close(self):
        self.win -= comission * order_value / 2
        self.position = 0
        self.win+=self.current_win - self.reward_tr_given
        self.reward_tr_given = 0
        self.current_win = 0
        self.closed_trades_since_last_log+=1
        
        
  def entry(self):
        self.entry_price = self.candles[self.current_step].c
        self.win -= comission * order_value / 2

  def step(self, action):
    
    if action == 0:
        if self.position != 0:
            self.close()
    
    if action == 1:
      #short
      if self.position == 1:
        self.close()

      if self.position == -1:
        pass
      else:
        self.position = -1
        self.entry()
        
    if action == 2:
      #long
      if self.position == -1:
        self.close()

      if self.position == 1:
        pass
      else:
        self.position = 1
        self.entry()
        
    self.current_step += 1
    if self.position != 0:
      current_price = self.candles[self.current_step].c
      entry = self.entry_price
      diff = (current_price - entry) / entry * order_value

      if self.position == 1:
        self.current_win = diff
      if self.position == -1:
        self.current_win = -diff

        
    diff = self.current_win - self.reward_tr_given
    reward_inc = diff / soft_reward_inc
    self.reward_tr_given += reward_inc
    self.win += reward_inc
    
    reward_raw = self.win# + self.current_win  # sparse reward disabled
    reward = reward_raw - self.last_reward
    self.last_reward = reward_raw
    reward = max(min(reward, 10), -10)
    
    
    done = self.current_step == len(self.candles) -1
    
    if (self.current_step - self.startindex) % log_interval == 0:
        log_reward = reward_raw - self.reward_since_last_log
        log_reward = max(min(log_reward, 200), -200)
        self.reward_since_last_log = reward_raw 
        file2 = open(log_folder+"logs/r2_log.txt", "a")  
        file2.write(str(log_reward))
        file2.write("\n")
        file2.close()
        
        
        file2 = open(log_folder+"logs/num_trades_per_day.txt", "a")  
        file2.write(str(self.closed_trades_since_last_log))
        file2.write("\n")
        file2.close()
        
        
        self.closed_trades_since_last_log = 0
    
    obs = self._next_observation()
    return obs, reward, done


class DQNAgent:
    def __init__(self, model,
                 n_actions,
                 memory_size = 10000, 
                 optimizer = tf.keras.optimizers.Adam(0.0005), 
                 gamma = 0.99,
                 batch_size =32,
                 name = "dqn1",
                 target_model_sync = 1000,
                 exploration = 0.01
                ):
        self.exploration = exploration
        self.gamma = gamma
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.model = model
        self.name = name
        self.memory_size = memory_size
        self.optimizer = optimizer
        self.m1 = np.eye(self.n_actions, dtype="float32")
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model_sync = target_model_sync
   
        self.memory = deque(maxlen = self.memory_size)
      
    
    def copy_weights(self):
        self.target_model.set_weights(self.model.get_weights())
      
    def load_weights(self):
        self.model.load_weights(self.name)
    def save_weights(self):
        self.model.save_weights(self.name, overwrite = True)
        
    @tf.function(jit_compile = True)
    def model_call(self, x):
        x1, x2 = x
        return tf.math.argmax(self.model([x1,x2]), axis = 1)
    
    def select_actions(self, current_states, positions):
        num_inputs = len(current_states)

        if random.random() < self.exploration:
          return tf.random.uniform(shape=[num_inputs], minval=0, maxval=3, dtype=tf.int32)

        assert num_inputs % 8 == 0
        inc = int(num_inputs/8)

        self.tn = -inc
        def vfunc(v):
          self.tn+=inc
          values = current_states[self.tn:self.tn+inc], positions[self.tn:self.tn+inc]
          return values

        inp = (strategy.experimental_distribute_values_from_function(vfunc))
        ret = strategy.run(self.model_call, args = (inp,))
        ret = np.array([x.numpy().tolist() for x in ret.values]).flatten()
        return ret


        
    def observe_sasrt(self, state, action, next_state, reward, terminal):
        self.memory.append([state, action, reward, 1-int(terminal), next_state])
        
    @tf.function(jit_compile = True)
    def get_target_q(self, next_states, rewards, terminals):
        estimated_q_values_next = self.target_model(next_states)
        q_batch = tf.math.reduce_max(estimated_q_values_next, axis=1)
        target_q_values = q_batch * self.gamma * terminals + rewards
        return target_q_values
    
        
    @tf.function(jit_compile = False)
    def tstep(self, states, next_states, rewards, terminals, masks):
        target_q_values = self.get_target_q(next_states, rewards, terminals)
        
        with tf.GradientTape() as t:
            estimated_q_values = tf.math.reduce_sum(self.model(states, training=True) * masks, axis=1)
            loss = tf.keras.losses.mean_absolute_error(target_q_values, estimated_q_values)
        
        gradient = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        
        return loss, tf.reduce_mean(estimated_q_values)
    
    
    def data_get_func(self, _n):
        sarts_batch = random.sample(self.memory, self.batch_size)
        
        states = [x[0] for x in sarts_batch]
        states_1 = np.array([x[0] for x in states], dtype="float32")
        states_2 = np.array([x[1] for x in states], dtype="float32")
        
        actions = [x[1] for x in sarts_batch]
        rewards = np.array([x[2] for x in sarts_batch], dtype="float32")
        terminals = np.array([x[3] for x in sarts_batch], dtype="float32")
        
        next_states = [x[4] for x in sarts_batch]
        next_states_1 = np.array([x[0] for x in next_states], dtype="float32")
        next_states_2 = np.array([x[1] for x in next_states], dtype="float32")
        
        masks = self.m1[actions]

        return [states_1, states_2], [next_states_1, next_states_2], rewards, terminals, masks

    def update_parameters(self):
        self.total_steps_trained+=1
        if self.total_steps_trained % self.target_model_sync == 0:
            self.copy_weights()

        distributed_values = (strategy.experimental_distribute_values_from_function(self.data_get_func))
        return  strategy.reduce(tf.distribute.ReduceOp.MEAN, strategy.run(self.tstep, args = (distributed_values)), axis = None)
    
    
    def train(self, num_steps, envs, log_interval = 1000, warmup = 0, train_steps_per_step = 1):
        self.total_steps_trained = -1

        num_envs = len(envs)
        states = [x.reset(True) for x in envs]
        
        current_episode_reward_sum = 0
        times= deque(maxlen=10)
        start_time = time.time()
        
        self.longs = 0
        self.shorts = 0

        self.total_rewards = []
        self.losses = [0]
        self.q_v = [0]
        
        def save_current_run():
            self.save_weights()
            file = open(log_folder+"logs/loss_log.txt", "a")  
            #for loss in self.losses:
                        #file.write(str(loss))
                        #file.write("\n")
            file.write(str(np.mean(self.losses)))
            file.close()

            file = open(log_folder+"logs/qv_log.txt", "a")  
            #for qv in self.q_v:
                        #file.write(str(qv))
                        #file.write("\n")
            file.write(str(np.mean(self.q_v)))
            file.close()

            file = open(log_folder+"logs/rewards_log.txt", "a")  
            #for total_reward in self.total_rewards:
                        #file.write(str(total_reward))
                        #file.write("\n")
                    
            file.write(str(np.mean(self.total_rewards)))
            file.close()
            
    

            self.total_rewards = []
            self.losses = [0]
            self.q_v = [0]
        
        try:
            for i in range(num_steps):
                if i % log_interval == 0:
                    #progbar = tf.keras.utils.Progbar(log_interval, interval=0.05, stateful_metrics = ["reward sum", "t", "l/s"])
                    self.longs = 0
                    self.shorts = 0


                states_1 = np.array([x[0] for x in states])
                states_2 = np.array([x[1] for x in states])
                actions = self.select_actions(states_1, states_2)
                for action in actions:
                    if action == 1:
                        self.shorts+=1
                    elif action == 2:
                        self.longs+=1

                sasrt_pairs = []
                for index in range(num_envs):
                    sasrt_pairs.append([states[index], actions[index]]+[x for x in envs[index].step(actions[index])])

                next_states = [x[2] for x in sasrt_pairs]

                reward = [x[3] for x in sasrt_pairs]
                current_episode_reward_sum += np.sum(reward)

                self.total_rewards.extend(reward)

                for index, o in enumerate(sasrt_pairs):
                    #print(o)
                    if o[4] == True:
                        next_states[index] = envs[index].reset()
                    self.observe_sasrt(o[0], o[1], o[2], o[3], o[4])

                states = next_states
                if i > warmup:
                    for _ in range(train_steps_per_step):
                        loss, q = self.update_parameters()
                        self.losses.append(loss.numpy())
                        self.q_v.append(q.numpy())
                else:
                    loss, q = 0, 0

                end_time = time.time()
                elapsed = (end_time - start_time) * 1000
                times.append(elapsed)
                start_time = end_time


                if (i+1) % log_interval == 0:
                    save_current_run()
                    print("-----------")
                    print("l:", np.mean(self.losses[-train_steps_per_step:]))
                    print("q:", np.mean(self.q_v[-train_steps_per_step:]))
                    print("r:", np.mean(reward))
                    print("reward sum", current_episode_reward_sum)
                    print("l/s", (self.longs - self.shorts) / (1+self.longs+self.shorts))
                    print("t", np.mean(times))
                    print("-----------")

                #progbar.update(i%log_interval+1, values = 
                               #[("loss", np.mean(self.losses[-train_steps_per_step:])),
                                #("mean q", np.mean(self.q_v[-train_steps_per_step:])),
                                #("rewards", np.mean(reward)),
                                #("reward sum", current_episode_reward_sum),
                                #("l/s", (self.longs - self.shorts) / (1+self.longs+self.shorts)),
                                #("t", np.mean(times))])
        
        except KeyboardInterrupt:
            print("\n\nbreak!")
        
        save_current_run()
            


  # detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu='local')

  # instantiate a distribution strategy
strategy = tf.distribute.experimental.TPUStrategy(tpu)




class TransformerBlock(tf.keras.layers.Layer):
    embed_dim = 0
    num_heads = 0
    ff_dim = 0 
    rate=0
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate


        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        cfg = super().get_config()
        cfg.update()
        cfg.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return cfg  
    def call(self, q, k, training = False):
        attn_output = self.att(q, k)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(q + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P[::]

class Positions(tf.keras.layers.Layer):
    P = []
    d = 0
    seq_len = 0
    def __init__(self, seq_len, d, **kwargs):
        super(Positions, self).__init__()
        self.seq_len = seq_len
        self.d = d
        self.p = getPositionEncoding(seq_len, d)
        

    def call(self, x):
        return x + self.p

    def get_config(self):
        cfg = super().get_config()
        cfg.update()
        cfg.update({
            'p': self.p,
            'seq_len': self.seq_len,
            'd': self.d
        })
        return cfg  




tf.keras.backend.clear_session()

with strategy.scope():
    
  inputs_1 = tf.keras.layers.Input(shape = (seq_len, 6))
  inputs_pos = tf.keras.layers.Input(shape = (2))

  x = inputs_1

  x = tf.keras.layers.Dense(16,activation = "relu")(x)
  x = tf.keras.layers.Dense(16,activation = "relu")(x)

  x2 = tf.keras.layers.Conv1D(64, 3,activation="relu", padding="same")(x)
  x = tf.keras.layers.Concatenate()([x2,x])

  x = tf.keras.layers.Dense(32,activation = "relu")(x)

  x2 = tf.keras.layers.Conv1D(512, 21,activation="relu", padding="same")(x)
  x = tf.keras.layers.Concatenate()([x2,x])

  x = tf.keras.layers.Dense(512,activation = "relu")(x)
  x = tf.keras.layers.Dense(256,activation = "relu")(x)
  
  x = tf.keras.layers.LayerNormalization()(x)

  x = Positions(seq_len, x.shape[-1])(x)
  x = TransformerBlock(x.shape[2], 8, 256)(x,x)
  x = TransformerBlock(x.shape[2], 8, 256)(x,x)
  x = TransformerBlock(x.shape[2], 8, 256)(x,x)
  x = TransformerBlock(x.shape[2], 8, 256)(x,x)
  x = TransformerBlock(x.shape[2], 8, 256)(x,x)
  x = TransformerBlock(x.shape[2], 8, 256)(x,x)

  x_end = tf.keras.layers.Lambda(lambda x: x[:,-1])(x)
  x_end = tf.keras.layers.Reshape((1,x.shape[2]))(x_end)
  x = TransformerBlock(x.shape[2], 8, 256)(x_end,x)
  x = tf.keras.layers.Flatten()(x)

  x = tf.keras.layers.Concatenate()([inputs_pos, x])

  x = tf.keras.layers.Dense(512)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)

  x = tf.keras.layers.Dense(512)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)

  x = tf.keras.layers.Dense(512)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
          
  outputs = tf.keras.layers.Dense(3, activation = "linear", use_bias=False, dtype="float32")(x)
  model = tf.keras.Model([inputs_1,inputs_pos], outputs)
model.summary()






with strategy.scope():
  opt = tf.keras.optimizers.Adam(lr, clipvalue = 0.5)

agent = DQNAgent(
    model = model, 
    n_actions = 3, 
    memory_size = memory_size, 
    gamma=gamma,
    optimizer = opt,
    batch_size = batch_size, 
    target_model_sync = 1000,
    exploration = 0.02,
    name=log_folder+name+".h5")

if resume:
	print("loading weights...")
	agent.load_weights()




num_parallel = warmup_parallel
envs = [environment() for _ in range(num_parallel)]

print("warmup...")

n = int(warmup_steps)
agent.train(num_steps = n, envs = envs, warmup = n, log_interval = n, train_steps_per_step=1)

num_parallel = training_parallel
envs = [environment() for _ in range(num_parallel)]


print("training...")

n = 100000000
agent.train(num_steps = n, envs = envs, warmup = 0, log_interval = 1000, train_steps_per_step=1)
