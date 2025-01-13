import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pkl


# FILE = "BB/results0/"
FILE = "PID/results0/"


def combine_PPO():
    # Define the folder path where your files are located
    folder_path = 'PPO/'

    # Define variations for patient and number
    seeds = [f'seed{i:01d}' for i in range(0, 20)]
    patient_name = 'adolescent#001'

    traj = []
    for seed in seeds:
        filename = f'{patient_name}.csv'
        file_path = os.path.join(folder_path, seed, filename)
        obs = []
        acts = []
        if os.path.isfile(file_path):
            # Read the data from the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            traj_len = len(df['BG'])
            df['done'] = np.array([False] * traj_len, dtype=bool)
            df['done'].iloc[-1] = np.array([True], dtype=bool)

            # for i in range(traj_len):
            #     obs.append([df['BG'].values[i]])
            #     acts.append([df['Action'].values[i]])

            for i in range(traj_len):
                obs.append(np.array([df['BG'].values[i]], dtype=np.float32))
                acts.append(np.array([df['Action'].values[i]], dtype=np.float32))

            # obs = df['BG'].values
            # acts = df['Action'].values

            traj.append({'observations': np.asarray(obs, dtype=np.float32), 
                        'actions': np.asarray(acts, dtype=np.float32), 
                        'rewards': np.asarray(df['reward'].values, dtype=np.float32), 
                        'terminals': np.asarray(df['done'].values)})
    
    
    print('Shape of list:', len(traj))
    print('Shape of observations:', len(traj[0]['observations']))
    print('Done', traj[0]['observations'])

    with open('PPO_adolescent_01.pkl', 'wb') as f:
        pkl.dump(traj, f)

def combine_PID_IF():
    # Define the folder path where your files are located
    folder_path = 'PID-IF'

    # Define variations for patient and number
    seeds = [f'-{i:01d}' for i in range(0, 20)]
    patient_name = 'adolescent#001'

    traj = []
    for seed in seeds:
        filename = f'{patient_name}{seed}.csv'
        file_path = os.path.join(folder_path, filename)
        obs = []
        acts = []
        if os.path.isfile(file_path):
            # Read the data from the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            traj_len = len(df['BG'])
            df['done'] = np.array([False] * traj_len, dtype=bool)
            df['done'].iloc[-1] = np.array([True], dtype=bool)

            # for i in range(traj_len):
            #     obs.append([df['BG'].values[i]])
            #     acts.append([df['Action'].values[i]])

            for i in range(traj_len):
                obs.append(np.array([df['BG'].values[i]], dtype=np.float32))
                acts.append(np.array([df['Action'].values[i]], dtype=np.float32))

            # obs = df['BG'].values
            # acts = df['Action'].values

            traj.append({'observations': np.asarray(obs, dtype=np.float32), 
                        'actions': np.asarray(acts, dtype=np.float32), 
                        'rewards': np.asarray(df['reward'].values, dtype=np.float32), 
                        'terminals': np.asarray(df['done'].values)})
    
    
    print('Shape of list:', len(traj))
    print('Shape of observations:', len(traj[0]['observations']))
    print('Done', traj[0]['observations'])

    with open('PID_IF_adolescent_01.pkl', 'wb') as f:
        pkl.dump(traj, f)


def plot_PPO_dataset():

    # Define the folder path where your files are located
    folder_path = 'PPO/'

    # Define variations for patient and number
    seeds = [f'seed{i:01d}' for i in range(0, 19)]
    patients = ['adolescent', 'child', 'adult']
    numbers = [f'#{i:03d}' for i in range(1, 11)]

    patient_name = 'adolescent#001'

    # Iterate through all combinations of seeds, patient and number
    for seed in seeds:
        for patient in patients:
            for number in numbers:
                # Construct the file name based on the patient and number
                filename = f'{patient}{number}.csv'
                file_path = os.path.join(folder_path, seed,filename)

                # Check if the file exists
                if os.path.isfile(file_path):
                    # Read the data from the CSV file into a DataFrame
                    df = pd.read_csv(file_path)

                    # Convert the 'Time' column to a datetime object
                    #df['Time'] = pd.to_datetime(df['Time'])

                    # Plot the graph for each file
                    plt.figure(figsize=(10, 6))
                    #plt.plot(df['Time'], df['BG'], marker='*', linestyle='-')
                    #plt.xlabel('Time')
                    plt.plot(df['no'], df['BG'], marker='*', linestyle='-')
                    plt.xlabel('Number')
                    plt.ylabel('BG')
                    plt.title(f'Blood Glucose Over Time - {filename}')
                    plt.grid(True)
                    plt.xticks(rotation=45)  # Rotate x-axis labels for readability

                    # Show or save the plot for each file
                    # plt.show()  # Uncomment this line to display each graph
                    # To save each graph as an image file, you can use plt.savefig()

        # If you want to save the plots as image files or display all together, you can do so outside the loop
        plt.show()  # To display all graphs together


if __name__ == "__main__":
    plot_PPO_dataset()
    # combine_PPO()
    #combine_PID_IF()
