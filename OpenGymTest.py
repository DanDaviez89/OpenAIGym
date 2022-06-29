import os
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#setting up environment
environment_name = "CartPole-v0"
env = gym.make(environment_name)

#This is how many times our game is going to run
episodes = 5

#This will loop through however many games we set
for episode in range(1, episodes+1):
    #Reset the enviroment
    #env.reset() - reset the environment and obtain intial observations
    state = env.reset()

    #Setting up more varibles
    #Will see if the game round is done
    done = False

    #will keep track of the score
    score = 0 
    
    #While the game round is not done run this code
    while not done:
        #Render the enviroment 
        #env.render() - Visualise the environment
        env.render()

        #Generating a random action
        action = env.action_space.sample()

        #Pass our random action to the enviroment
        #env.step() - Apply an action to the environement
        n_state, reward, done, info = env.step(action)

        #This accumulates a reward
        score+=reward
    #Then we print out the results
    print('Episode:{} Score:{}'.format(episode, score))

#Close down the render frame
env.close()



#Train an RL model
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)

model.learn(total_timesteps = 20000)



#Save and Reload Model
#Set Save path
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')

#Save Model
model.save(PPO_path)

#delete model
del model

#Load
model = PPO.load('Training/Saved Models/PPO_model.zip', env=env)

#Evaluation
from stable_baselines3.common.evaluation import evaluate_policy

evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()



#Run the enviroment but using the trained model
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done: 
        print('Episode:{} Score:{}'.format(episode, score))
        break
env.close()



#Adding a callback to the training Stage
#Need to call this to allow callbacks
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

#Where we want to save our new best model
save_path = os.path.join('Training', 'Saved Models')

log_path = os.path.join('Training', 'Logs')

env = gym.make(environment_name)

env = DummyVecEnv([lambda: env])

#Stop training once we pass a certain reward threshold
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)

#This is the callback that is going to be triggered after each training run
eval_callback = EvalCallback(env, #pass through environment
                             callback_on_new_best=stop_callback, #every time theres a new best model, its going to run the stop call back
                                                                 #If the stop callback realizes that the reward threshold is above 200 then its going to stop training
                             eval_freq=10000, #How often we get our evaluation callback
                             best_model_save_path=save_path, #Specify best model, save our best model to save path
                             verbose=1)

#Create a new model
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

#Teach new model (number of steps its going to run, setting our call back)
#Will callback every 10000 steps and save the best new model if it has one and check if its passed that average reward
model.learn(total_timesteps=20000, callback=eval_callback)

#setting model path
model_path = os.path.join('Training', 'Saved Models', 'best_model')

#Setting new model
model = PPO.load(model_path, env=env)

#testing new model
evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()



#Changing Policies
net_arch=[dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]

model = PPO('MlpPolicy', env, verbose = 1, policy_kwargs={'net_arch': net_arch})

model.learn(total_timesteps=20000, callback=eval_callback)



#Using an Alternate Algorithm
#Train a DQN algorithm instead PPO
from stable_baselines3 import DQN

#Set up DQN model
model = DQN('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

#Teach network using our callback
model.learn(total_timesteps=20000, callback=eval_callback)

#Save path
dqn_path = os.path.join('Training', 'Saved Models', 'DQN_model')

#Save DQN model
model.save(dqn_path)

#Load model
model = DQN.load(dqn_path, env=env)

#Evaluate Model
evaluate_policy(model, env, n_eval_episodes=10, render=True)

#Close Model
env.close()