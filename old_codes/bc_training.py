
import torch as T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import pickle as pkl
from examples.network.bc_model import BC
import numpy as np
import gym
from gym.envs.registration import register
import simglucose
import wandb

class bc_dataset(Dataset):
    def __init__(self, demos):
        self.data = []
        for i in range(len(demos)):
            traj = demos[i]
            for k, state in enumerate(traj['observations']):
                action = traj['actions'][k]
                self.data.append((state,action))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s, a = self.data[index]
        return s, a


def load_data():
    data = []
    SAVE_PATH = '/home/berk/VS_Project/simglucose/examples/trajectories/DATA_eps_11-2022-11-21 20:20:00.pkl'
    with open(SAVE_PATH, 'rb') as handle:
        data = pkl.load(handle)
    return data

def main():
    memory = []
    memory = load_data()

    wandb.init(name='Behavior Cloning', 
           project='Simglucose',
           notes='Behavior Cloning dataset with 11 trajectory', 
           tags=['SimBG Dataset', 'Test Run'])

    register(
        id='simglucose-adult-v1',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': 'adult#001',}
    )

    env = gym.make('simglucose-adult-v1')

    obs_dim = memory[0]['observations'].shape[1]
    action_dim = memory[0]['actions'].shape[1]
    hidden_dim = 128
    batch_size = 32

    epochs = 5000
    n_traj = len(memory)
    horizon = memory[0]['actions'].shape[0]

    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    Policy = BC(obs_dim, hidden_dim, action_dim).to(device)
    dataset = bc_dataset(memory)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    optimizer = optim.Adam(Policy.parameters(), lr=1e-3, weight_decay=0.0001)
    criterion = nn.MSELoss()

    ## Training stage ##
    Policy.train()
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(loader):
            state, action = data
            state = state.to(device)
            action = action.to(device)
            optimizer.zero_grad()
            action_pred = Policy(state)
            loss = criterion(action_pred, action.cuda().float())
            running_loss +=loss
            # print("Loss: ", loss)
            loss.backward()
            if i%200 == 199:
                print("Epoch:%d -- Batch:%d -- Loss:%f"%(epoch, i+1, running_loss/n_traj))
                running_loss = 0

            optimizer.step()

        wandb.log({
        "Epoch": epoch,
        "Running Loss": running_loss,
        "Mean Loss": T.mean(running_loss)})

    print("Training finished..:")

    ## Evaluation stage ##
    Policy.eval()
    Policy.cpu()
    returns = []
    
    for i  in range(10):
        print("Evaluation:%d"%i)
        obs = env.reset()
        obs = np.array(obs.CGM)
        done = False
        totalr = 0
        steps = 0
        while not done:
            env.render(mode='human')
            obs = T.from_numpy(obs).unsqueeze(dim=0).float()
            action_p = Policy(obs).detach().numpy()
            obs, reward, done, _ = env.step(action_p)
            obs = np.array(obs.CGM)
            totalr += reward
            steps +=1
            if steps %288 == 0: print("Step/Horizon:%d/%d"%(steps,horizon))
            if steps >= horizon: break
            returns.append(totalr)

    print("Returns:", returns)
    print("Mean of Returns:", np.mean(returns))
    print("Std of Returns:", np.std(returns))


    model = T.save(Policy, '/home/berk/VS_Project/simglucose/models/model_bc.h5')
    artifact = wandb.Artifact('Torch-BC',type='model')
    artifact.add_file('/home/berk/VS_Project/simglucose/models/model_bc.h5')


if __name__ == "__main__":
    main()
    