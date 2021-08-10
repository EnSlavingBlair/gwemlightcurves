import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import corner 


Types = ['BNS_alsing','BNS_farrow','BNS_equal_alsing','BNS_equal_farrow','BNS_uniform','NSBH_uniform','NSBH_zhu','BNS_chirp_q']
Types = ['BNS_alsing', 'BNS_farrow']
Types = ['NSBH_zhu', 'NSBH_LRR']
#Types = ['NSBH_LRR']
#Types = ['BNS_alsing']

frac_0s = []

Type_last = 'none'
for Type in Types:
    #mej_theta_data = np.loadtxt(f'./mej_theta_data/mej_theta_data_{Type}.txt')
    #mej_data, thetas = mej_theta_data[:,0], mej_theta_data[:,1]
   
    #initial_params = np.loadtxt(f'./corner_data/corner_data_{Type}.txt')
    initial_params = np.loadtxt(f'./corner_data/NSBH_test/corner_data_{Type}.txt')
    #all_m1s, all_m2s, all_mchirps, all_qs, all_vej, all_mej_data, all_wind_mej, all_dyn_mej, all_thetas
    #m1, m2, mchirp, mej, wind_mej, dyn_mej, thetas
    mej_initial = initial_params[:,5]
    theta_initial = initial_params[:,8]
    initial_params = initial_params[:,(0,1,2,6,7)]
  
    print(initial_params.shape)
    
    #folder_dir = f'./lightcurves_parallel/{Type}/'
    #folder_dir = f'./lightcurves2/{Type}/'
    folder_dir = f'./lightcurves_parallel/phi45_updated/{Type}/'
    ns_dirs = os.listdir(f'{folder_dir}')
    print(len(ns_dirs))
    #ns_dirs = ns_dirs[:100]
    
    #n_mej = len(ns_dirs)
    #mej_data, thetas = mej_data[:n_mej], thetas[:n_mej]
    #initial_params = initial_params[:n_mej]


    nsns_dict = {}
    nsbh_dict = {}
    bands = ['t','u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K', 'mej', 'theta', 'phi']

    for band in bands:
        nsns_dict[band], nsbh_dict[band] = [],[]

    count = 0
    for ns_dir in ns_dirs:   
        count+=1
        with open (f'./{folder_dir}/{ns_dir}','rb') as f:
            data = pickle.load(f)
        if count%1000 == 0:
            print(f'{count} samples loaded')
        for ii,band in enumerate(bands):
            nsns_dict[band].append(data[:,ii])

    f,axes=plt.subplots(ncols=5,nrows=2,figsize=(35,15),sharey='row')
    plt.rcParams['figure.dpi'] = 200

    
    mej_data = []
    theta_data = []
    mejs = nsns_dict['mej']
    thetas = nsns_dict['theta']
    
 
    for mej_vals in mejs:
        mej_data.append(mej_vals[0])
    for theta_vals in thetas:
        theta_data.append(theta_vals[0])
   
    #print(mej_initial)
    #print(theta_initial)
    idx_sort = []
    for (mej, theta) in zip(mej_data, theta_data):
        #print(mej,theta)
        #idx_m = np.argwhere(mej_initial == mej)
        #idx_t = np.argwhere(theta_initial == theta)
        idx_m = np.argwhere((np.abs(mej_initial-mej)) <= 1e-6)
        #print(idx_m)
        idx_t = np.argwhere((np.abs(theta_initial-theta)) <= 1e-6)
        for mm in idx_m:
            for tt in idx_t:
                if mm == tt:
                    idx_sort.append(mm)
    #print(idx_sort)
    #print(initial_params.shape)         
    initial_params_sorted = initial_params[idx_sort]
    #print(initial_params_sorted.shape)
    initial_params_sorted = initial_params_sorted[:,0,:]
    
    mej_data = np.array(mej_data)[idx_sort]
    theta_data = np.array(theta_data)[idx_sort]
    #mej_data = mej_data[:,0,:]
    #theta_data = theta_data[:,0,:]

    print(f'Initializing {Type}') 

    N_lc = len(mej_data)
    idx_nonzero = np.argwhere(np.array(mej_data) >= 1e-9)

    frac_mej0 = (N_lc - len(idx_nonzero))/N_lc
    print(f'Fraction of Kilonovae with mej = 0: {frac_mej0}')
    frac_0s.append(frac_mej0)
    
    mej_data = mej_data[idx_nonzero]
    theta_data = theta_data[idx_nonzero]
    mej_data = mej_data[:,0,:]
    theta_data = theta_data[:,0,:]

    initial_params_sorted = initial_params_sorted[idx_nonzero]
    initial_params_sorted = initial_params_sorted[:,0,:]
    t = np.array(nsns_dict['t'])
    #first sort then cut out mej = 0
    t = t[idx_sort]
    t = t[:,0,:]
    #t = t[idx_nonzero]
    t = t[idx_nonzero][:]
    t = t[:,0,:]
    shape1, shape2 = t.shape[0], t.shape[1]
    t = np.reshape(t, (shape1*shape2))


    n_tsteps = 150
    for (i,j,band) in zip([0,0,0,0,0,1,1,1,1],[0,1,2,3,4,0,1,2,3],bands[1:10]):
        nsns = np.array(nsns_dict[band])
        nsns = nsns[idx_sort]
        nsns = nsns[:,0,:]
        #nsns = nsns[idx_nonzero]
        nsns = nsns[idx_nonzero]
        nsns = nsns[:,0,:]
        t_bins = t[0:n_tsteps]
        bins = np.linspace(-21, 0, 50)
        
        shape1, shape2 = nsns.shape[0], nsns.shape[1]
        nsns_1D = np.reshape(nsns, (shape1*shape2))        
       
 
        hist2d_1, xedges, yedges = np.histogram2d(t, nsns_1D, bins = (t_bins,bins))
        X, Y = np.meshgrid(xedges, yedges)
        hist2d_1[hist2d_1 == 0] = np.nan

        im = axes[i][j].pcolormesh(X, Y, hist2d_1.T, shading = 'auto', cmap='viridis',alpha=0.7)
        print(str(band)+' complete') 
       
        p_list = []
        for t_val in t_bins:
            p_list.append(nsns_1D[t == t_val])
        p_list = np.array(p_list)

        n_lc = p_list.shape[1]
 
        peak_mag = []
        for lc in nsns:
            peak_mag.append(np.min(lc[0:n_tsteps]))
        if band == 'r':
            r_peak = peak_mag
        if band == 'K': 
            K_peak = peak_mag

 
        per_list = [0, 10, 50, 90, 100]
        for percentile in per_list:
            per = np.nanpercentile(p_list, percentile, axis=1)
            lc_diff = []
            for lc in nsns:
                lc_diff.append(np.sum(np.abs(lc[0:n_tsteps]-per)))
            idx = np.argmin(lc_diff)
            #print('idx: ' + str(idx))    
            lc = nsns[idx]
            mej, theta = mej_data[idx], theta_data[idx]
            #axes[i][j].plot(t_bins, lc[0:n_tsteps], linestyle = '--', label = f'{percentile}th percentile: mej={mej}, theta={theta}')
            axes[i][j].plot(t_bins, lc[0:n_tsteps], linestyle = '--', label = f'{percentile}th percentile: mej={round(float(mej),6)}, theta={round(float(theta),0)}') 
            if Type_last == Type:
                if idx_last != idx_last:
                    print(f'different percentiles between bands!!')
            Type_last = Type
            idx_last = Type
                        

        #plot 10th, 50th, 90th percentiles
        #axes[i][j].plot(t_bins, np.nanpercentile(p_list, 50, axis=1), c='k',linestyle='--',label=f'{Type}')
        #axes[i][j].plot(t_bins, np.nanpercentile(p_list, 90, axis=1),'k--')
        #axes[i][j].plot(t_bins, np.nanpercentile(p_list, 10, axis=1),'k--')
 
        if band == 'K':
            cb_ax = f.add_axes([0.94, 0.14, 0.023, 0.7])
            cb = f.colorbar(im, cax = cb_ax, ticks=[])
            cb.set_label(label='NSNS',size=30)

        axes[i][j].set_ylim([0,-21])
        axes[i][j].text(1,-19,f'{band}',size=30)
        axes[i][j].tick_params(axis='x', labelsize=30)
        axes[i][j].tick_params(axis='y', labelsize=30)

    f.text(0.5,0.05,'Time [days]',size=30)
    axes[0][0].set_ylabel('$M_{AB}$',size=30)
    axes[1][0].set_ylabel('$M_{AB}$',size=30)

    axes[-1, -1].axis('off')

    h1, l1 = axes[0][0].get_legend_handles_labels()
    h2, l2 = axes[1][1].get_legend_handles_labels()

    #Make the legend
    legend = axes[-1][-1].legend(h1, l1,  bbox_to_anchor=(0,1,1.0,-0.15), loc=9,
               ncol=1,prop={'size': 18},fancybox=True,frameon=True)

    frame = legend.get_frame()
    frame.set_color('skyblue')
    plt.savefig(f'./heatmaps_corner/heatmap_{Type}.pdf',bbox_inches='tight')

    #N_lc = len(mej_data)
    #idx_nonzero = np.argwhere(np.array(mej_data) >= 1e-9)
    
    #frac_mej0 = (N_lc - len(idx_nonzero))/N_lc
    #print(f'Fraction of Kilonovae with mej = 0: {frac_mej0}')
    #frac_0s.append(frac_mej0)

    #corner_plot = np.column_stack((initial_params[:,(0,1,2)], np.log10(initial_params[:,(3,4)]), np.log10(mej_data), theta_data, r_peak, K_peak))
    corner_plot = np.column_stack((initial_params_sorted[:,(0,1,2)], np.log10(initial_params_sorted[:,(3,4)]), theta_data))
    #print(corner_plot.shape)
    #corner_plot = corner_plot[idx_nonzero,:]
    #corner_plot = corner_plot[:,0,:]
    corner_plot = np.column_stack((corner_plot[:,(0,1,2,3,4)], np.log10(mej_data), corner_plot[:,5], r_peak, K_peak))
    #print(corner_plot.shape)
    corner.corner(corner_plot, labels=['m1','m2','mchirp','log(wind_mej)','log(dyn_mej)', 'log(mej)', 'theta', 'peak r mag', 'peak K mag'], range=([1.1,2.8],[1.1,2.8],1,[-6,.5],[-6,.5],[-6,.5],1,[-11,-18],[-11,-18]))
    #corner.corner(corner_plot, labels=['m1','m2','mchirp','log(wind_mej)','log(dyn_mej)', 'log(mej)', 'theta', 'peak r mag', 'peak K mag'], range=([1.1,2.8],[1.1,2.8],1,1,[0,.15],[0,.15],1,[-11,-18],[-11,-18]))
    #corner.corner(corner_plot, labels=['m1','m2','mchirp','mej','vej', 'dyn_mej', 'wind_mej'], range=([1.25,1.6],[1.25,1.6],1,[-4,0],1,[-4,0],[-4,0]))

    plt.savefig(f'./heatmaps_corner/corner_updated_{Type}.pdf')
    plt.close()

    print(f'{Type} complete')


print('Fraction of Kilonovae with mej = 0 by Type:')
print(Types)
print(frac_0s)
