import gym

env = gym.make('Breakout-v0', render_mode='human')
env.reset()

for i in range(1):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("this is obs: ",obs)
    print("this is reward: ", reward)
    print("this is done: ", done)
    print("this is info: ", info)
