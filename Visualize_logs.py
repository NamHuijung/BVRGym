import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_log_data(log_dir, episode_numbers):
    log_files = [f for f in os.listdir(log_dir) if f.endswith('_log.json')]
    all_data = []

    for log_file in log_files:
        episode_number = int(log_file.split('_')[1])  # Assuming the log file name is like 'episode_1_log.json'
        if episode_number in episode_numbers:
            with open(os.path.join(log_dir, log_file), 'r') as f:
                data = json.load(f)
                all_data.append(data)
    
    return all_data

def process_log_data(log_data):
    episodes = {}
    for data in log_data:
        for entry in data:
            sim_time = entry['sim_time']

            if sim_time not in episodes:
                episodes[sim_time] = {}

            for key, value in entry.items():
                if key not in ['cpu_id', 'sim_time']:
                    if key not in episodes[sim_time]:
                        episodes[sim_time][key] = []
                    episodes[sim_time][key].append(value)
    
    return episodes

def compute_statistics(episodes):
    sim_times = sorted(episodes.keys())
    stats = {}

    for sim_time in sim_times:
        for key, values in episodes[sim_time].items():
            if key not in stats:
                stats[key] = {'mean': [], 'std': []}
            stats[key]['mean'].append(np.mean(values))
            stats[key]['std'].append(np.std(values))

    return sim_times, stats

def smooth_data(data, alpha):
    ema = [data[0]]  # EMA starts with the first data point
    for point in data[1:]:
        ema.append(alpha * point + (1 - alpha) * ema[-1])
    return np.array(ema)

def plot_statistics(sim_times, stats, output_dir, smoothing_factor):
    for key, stat in stats.items():
        mean_values = np.array(stat['mean'])
        std_values = np.array(stat['std'])

        smoothed_mean = smooth_data(mean_values, smoothing_factor)
        smoothed_std = smooth_data(std_values, smoothing_factor)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sim_times, smoothed_mean, label=f'Smoothed Mean {key}')
        plt.fill_between(sim_times, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, color='b', alpha=0.2, label='±1 Std Dev')

        plt.xlabel('Simulation Time (s)')
        plt.ylabel(key)
        plt.title(f'{key} over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{key}_plot.png'))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_dir', '--log_dir', type=str, required=True, help='Directory containing log files')
    parser.add_argument('-output_dir', '--output_dir', type=str, required=True, help='Directory to save plots')
    parser.add_argument('-episodes', type=int, nargs='+', required=True, help='Episode numbers to include in the analysis')
    parser.add_argument('-smoothing_factor', type=float, default=0.6, help='Smoothing factor for exponential moving average')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_data = load_log_data(args.log_dir, args.episodes)
    episodes = process_log_data(log_data)
    sim_times, stats = compute_statistics(episodes)
    plot_statistics(sim_times, stats, args.output_dir, args.smoothing_factor)
