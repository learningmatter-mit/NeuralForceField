import numpy as np
import torch
import matplotlib.pyplot as plt

def make_plot(key_pairs, results, targets, target_dic=None):
    
    all_keys = np.array(key_pairs).reshape(-1)
    units = dict()
    for key in all_keys:
        if "grad" in key:
            units[key] = r'kcal/mol/$\AA$'
        else:
            units[key] = 'kcal/mol'

    if target_dic is None:
        target_dic = {key: key for key in all_keys}

    for i in range(len(key_pairs)):

        fig, ax_fig = plt.subplots(1, 2, figsize=(12, 6))

        for ax, key in zip(ax_fig, key_pairs[i]):

            if key not in targets.keys():
                targ_key = correspondence_keys[key]
            else:
                targ_key = key
                
            pred = (torch.cat(results[key])).reshape(-1).cpu().detach().numpy()
            target_key = target_dic[key]
            try:
                targ = (torch.cat(targets[targ_key])).reshape(-1).cpu().detach().numpy()
            except:
                targ = (torch.stack(targets[targ_key])).reshape(-1).cpu().detach().numpy()


            ax.scatter(pred, targ, color='#ff7f0e', alpha=0.3)

            mae = np.mean(abs(pred-targ))
            if "grad" in key:
                these_units = r"kcal/mol/$\AA$"
            else:
                these_units = r"kcal/mol"
            plt.text(0.1, 0.75, "MAE = {} {}".format(str(round(mae, 1)), these_units),
                     transform=ax.transAxes, fontsize=14)

            lim_min = min(np.min(pred), np.min(targ)) * 1.1
            lim_max = max(np.max(pred), np.max(targ)) * 1.1

            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.set_aspect('equal')

            ax.plot((lim_min, lim_max),
                    (lim_min, lim_max),
                    color='#000000',
                    zorder=-1,
                    linewidth=0.5)

            ax.set_title(key.upper(), fontsize=14)
            ax.set_xlabel('predicted %s (%s)' % (key, units[key]), fontsize=12)
            ax.set_ylabel('target %s (%s)' % (key, units[key]), fontsize=12)

