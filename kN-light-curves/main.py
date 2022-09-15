import numpy as np
from astropy.table import Table
from distutils.spawn import find_executable
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves import lightcurve_utils
import h5py

### My version of the code to create kilonova light curves
### Modified version of code from :
### https://gwemlightcurves.github.io/examples/index.html#generating-light-curves

def KNTable_read_samples_TSB(filename_samples, Nsamples = 100):
    import os
    if not os.path.isfile(filename_samples):
        raise ValueError("Sample file supplied does not exist")
    if "hdf" in filename_samples:
        samples_out = h5py.File(filename_samples, 'r')
        samples_out = samples_out['lalinference']
        data_out = Table(samples_out)
        data_out['q'] = data_out['m1_source'] / data_out['m2_source']
        data_out['mchirp'] = (data_out['m1_source'] * data_out['m2_source']) ** (3. / 5.) / (
                    data_out['m1_source'] + data_out['m2_source']) ** (1. / 5.)
        data_out['theta'] = data_out['iota']
        idx = np.where(data_out['theta'] > 90.)[0]
        data_out['theta'][idx] = 180 - data_out['theta'][idx]
        data_out["eta"] = lightcurve_utils.q2eta(data_out["q"])
        data_out["m1"], data_out["m2"] = lightcurve_utils.mc2ms(data_out["mchirp"], data_out["eta"])
        data_out['q'] = 1.0 / data_out['q']
    else:
        data_out = Table.read(filename_samples, format='ascii')
        if 'mass_1_source' in list(data_out.columns):
            data_out['m1'] = data_out['mass_1_source']
            print('setting m1 to m1_source')
        if 'mass_2_source' in list(data_out.columns):
            data_out['m2'] = data_out['mass_2_source']
            print('setting m2 to m2_source')
        if 'm1_detector_frame_Msun' in list(data_out.columns):
            data_out['m1'] = data_out['m1_detector_frame_Msun']
            print('setting m1 to m1_source')
        if 'm2_detector_frame_Msun' in list(data_out.columns):
            data_out['m2'] = data_out['m2_detector_frame_Msun']
            print('setting m2 to m2_source')
        if 'dlam_tilde' in list(data_out.columns):
            data_out['dlambdat'] = data_out['dlam_tilde']
            print('setting dlambdat to dlam_tilde')
        if 'lam_tilde' in list(data_out.columns):
            data_out['lambdat'] = data_out['lam_tilde']
            print('setting lambdat to lam_tilde')
        if 'delta_lambda_tilde' in list(data_out.columns):
            data_out['dlambdat'] = data_out['delta_lambda_tilde']
            print('setting dlambdat to delta_lambda_tilde')
        if 'lambda_tilde' in list(data_out.columns):
            data_out['lambdat'] = data_out['lambda_tilde']
            print('setting lambdat to lambda_tilde')
        if 'm1' not in list(data_out.columns):
            eta = lightcurve_utils.q2eta(data_out['mass_ratio'])
            m1, m2 = lightcurve_utils.mc2ms(data_out["chirp_mass"], eta)
            data_out['m1'] = m1
            data_out['m2'] = m2
        data_out['mchirp'], data_out['eta'], data_out['q'] = lightcurve_utils.ms2mc(data_out['m1'], data_out['m2'])
        data_out['q'] = 1.0 / data_out['q']
        if ('spin1' in data_out.keys()) and ('spin2' in data_out.keys()):
            data_out['chi_eff'] = ((data_out['m1'] * data_out['spin1'] + data_out['m2'] * data_out['spin2']) / (
                        data_out['m1'] + data_out['m2']))
        elif ('chi1' in data_out.keys()) and ('chi2' in data_out.keys()):
            data_out['chi_eff'] = ((data_out['m1'] * data_out['chi1'] + data_out['m2'] * data_out['chi2']) / (
                        data_out['m1'] + data_out['m2']))
        else:
            try:
                data_out['chi_eff'] = 0.0
            except KeyError:
                print("WARNING: KeyError. Could not find 'chi_eff' in table")
                pass
        if "luminosity_distance_Mpc" in data_out.keys():
            data_out["dist"] = data_out["luminosity_distance_Mpc"]
        elif "luminosity_distance" in data_out.keys():
            data_out["dist"] = data_out["luminosity_distance"]
    data_out = KNTable(data_out)
    data_out = data_out.downsample(Nsamples)
    return data_out

def get_eos_list(TOV):
    """
    Populates lists of available EOSs for each set of TOV solvers
    """
    import os
    if TOV not in ['Monica', 'Wolfgang', 'lalsim']:
        raise ValueError('You have provided a TOV for which we have no data and therefore cannot calculate the radius.')
    try:
        path = find_executable('/gwemlightcurves-master/input/'+TOV+'/ap4_mr.dat')
        path = path[:-10]
    except:
       raise ValueError('Check to make sure EOS mass-radius tables have been installed correctly (try `which ap4_mr.dat`)')
    if TOV == 'Monica':
        EOS_List=[file_name[:-7] for file_name in os.listdir(path) if file_name.endswith("_mr.dat") and 'lalsim' not in file_name]
    if TOV == 'Wolfgang':
        EOS_List=[file_name[:-10] for file_name in os.listdir(path) if file_name.endswith("seq")]
    if TOV == 'lalsim':
        EOS_List=[file_name[:-14] for file_name in os.listdir(path) if file_name.endswith("lalsim_mr.dat")]
    return EOS_List

def construct_eos_from_polytrope(eos_name):
    """
    Uses lalsimulation to read polytrope parameters from table
    """
    import lalsimulation as lalsim
    from astropy.io import ascii
    polytrope_table=np.genfromtxt(find_executable('polytrope_table.dat'), dtype=("|S10", '<f8','<f8','<f8','<f8'), names=True)
    #convert all eos names to lower case
    for i in range(0,len(polytrope_table['eos'])):
        polytrope_table['eos'][i]=polytrope_table['eos'][i].lower()
    #convert logp from cgs to si
    for i in range(0, len(polytrope_table['logP1'])):
        polytrope_table['logP1'][i]=np.log10(10**(polytrope_table['logP1'][i])*0.1)
    eos_indx=np.where(polytrope_table['eos']==eos_name.encode('utf-8'))[0][0]
    eos=lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(polytrope_table['logP1'][eos_indx], polytrope_table['gamma1'][eos_indx], polytrope_table['gamma2'][eos_indx], polytrope_table['gamma3'][eos_indx])
    fam=lalsim.CreateSimNeutronStarFamily(eos)
    return eos, fam

def calc_radius_TSB(t, EOS, TOV, polytrope=False):
    """
    """
    if TOV not in ['Monica', 'Wolfgang', 'lalsim']:
        raise ValueError('You have provided a TOV for which we have no data and therefore cannot calculate the radius.')
    if EOS not in get_eos_list(TOV):
        raise ValueError('You have provided a EOS for which we have no data and therefore cannot calculate the radius.')
    if TOV == 'Monica':
        import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
        import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
        MassRadiusBaryMassTable = Table.read(find_executable(EOS + '_mr.dat'), format='ascii')
        radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'])
        # after obtaining the radius_of_mass constants we now can either take values directly from table or use pre calculated spline to extrapolate the values
        # also radius is in km in table. need to convert to SI (i.e. meters)
        t['r1'] = et.values_from_table(t['m1'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const) * 10 ** 3
        t['r2'] = et.values_from_table(t['m2'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const) * 10 ** 3
    elif TOV == 'Wolfgang':
        import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
        import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
        try:
            import lal
            G = lal.G_SI;
            c = lal.C_SI;
            msun = lal.MSUN_SI
        except:
            import astropy.units as u
            import astropy.constants as C
            G = C.G.value;
            c = C.c.value;
            msun = u.M_sun.to(u.kg)
        MassRadiusBaryMassTable = Table.read(find_executable(EOS + '.tidal.seq'), format='ascii')
        radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'])
        unit_conversion = (msun * G / c ** 2)
        t['r1'] = et.values_from_table(t['m1'], MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'], radius_of_mass_const) * unit_conversion
        t['r2'] = et.values_from_table(t['m2'], MassRadiusBaryMassTable['grav_mass'], MassRadiusBaryMassTable['Circumferential_radius'], radius_of_mass_const) * unit_conversion
    elif TOV == 'lalsim':
        import lalsimulation as lalsim
        if polytrope == True:
            try:
                import lal
                G = lal.G_SI;
                c = lal.C_SI;
                msun = lal.MSUN_SI
            except:
                import astropy.units as u
                import astropy.constants as C
                G = C.G.value;
                c = C.c.value;
                msun = u.M_sun.to(u.kg)
            ns_eos, eos_fam = construct_eos_from_polytrope(EOS)
            t['r1'] = np.vectorize(lalsim.SimNeutronStarRadius)(t["m1"] * msun, eos_fam)
            t['r2'] = np.vectorize(lalsim.SimNeutronStarRadius)(t["m2"] * msun, eos_fam)
        else:
            import gwemlightcurves.EOS.TOV.Monica.MonotonicSpline as ms
            import gwemlightcurves.EOS.TOV.Monica.eos_tools as et
            MassRadiusBaryMassTable = Table.read(find_executable(EOS + '_lalsim_mr.dat'), format='ascii')
            radius_of_mass_const = ms.interpolate(MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'])
            # after obtaining the radius_of_mass constants we now can either take values directly from table or use pre calculated spline to extrapolate the values
            # also radius is in km in table. need to convert to SI (i.e. meters)
            t['r1'] = et.values_from_table(t['m1'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const) * 10 ** 3
            t['r2'] = et.values_from_table(t['m2'], MassRadiusBaryMassTable['mass'], MassRadiusBaryMassTable['radius'], radius_of_mass_const) * 10 ** 3
    return t

def plot_mag_panels_TSB(table_dict, distance, filts=["g", "r", "i", "z", "y", "J", "H", "K"], magidxs=[0, 1, 2, 3, 4, 5, 6, 7, 8], figsize=(20, 28)):
    """
    This allows us to take the lightcurves from the KNModels samples table and plot it
    using a supplied set of filters. Default: filts=["g","r","i","z","y","J","H","K"]
    """
    # get legend determines the names to add to legend based on KN model
    def get_legend(model):
        if model == "DiUj2017":
            legend_name = "Dietrich and Ujevic (2017)"
        if model == "KaKy2016":
            legend_name = "Kawaguchi et al. (2016)"
        elif model == "Me2017":
            legend_name = "Metzger (2017)"
        elif model == "SmCh2017":
            legend_name = "Smartt et al. (2017)"
        elif model == "WoKo2017":
            legend_name = "Wollaeger et al. (2017)"
        elif model == "BaKa2016":
            legend_name = "Barnes et al. (2016)"
        elif model == "Ka2017":
            legend_name = "Kasen (2017)"
        elif model == "RoFe2017":
            legend_name = "Rosswog et al. (2017)"
        return legend_name
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.size': 16})
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    # Initialize variables and arrays
    models = table_dict.keys()
    colors_names = cm.rainbow(np.linspace(0, 1, len(models)))
    tt = np.arange(table_dict[list(models)[0]]['tini'][0], table_dict[list(models)[0]]['tmax'][0] + table_dict[list(models)[0]]['dt'][0], table_dict[list(models)[0]]['dt'][0])
    # Initialize plot
    plt.figure(figsize=figsize)
    cnt = 0
    for filt, magidx in zip(filts, magidxs):
        cnt = cnt + 1
        vals = "%d%d%d" % (len(filts), 1, cnt)
        if cnt == 1:
            ax1 = plt.subplot(eval(vals))
        else:
            ax2 = plt.subplot(eval(vals), sharex=ax1, sharey=ax1)
        for ii, model in enumerate(models):
            legend_name = get_legend(model)
            magmed = np.median(table_dict[model]["mag_%s" % filt], axis=0)
            magmax = np.max(table_dict[model]["mag_%s" % filt], axis=0)
            magmin = np.min(table_dict[model]["mag_%s" % filt], axis=0)
            plt.plot(tt, magmed, '--', c=colors_names[ii], linewidth=2, label=legend_name)
            plt.fill_between(tt, magmin, magmax, facecolor=colors_names[ii], alpha=0.2)
        plt.ylabel('%s' % filt, fontsize=48, rotation=0, labelpad=40)
        plt.xlim([0.0, 14.0])
        plt.ylim([-18.0, -10.0])
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        if cnt == 1:
            ax1.set_yticks([-18, -16, -14, -12, -10])
            plt.setp(ax1.get_xticklabels(), visible=False)
            l = plt.legend(loc="upper right", prop={'size': 24}, numpoints=1, shadow=True, fancybox=True)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            ax3 = ax1.twinx()  # mirror them
            ax3.set_yticks([16, 12, 8, 4, 0])
            app = np.array([-18, -16, -14, -12, -10]) + np.floor(5 * (np.log10(distance * 1e6) - 1))
            ax3.set_yticklabels(app.astype(int))
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
        else:
            ax4 = ax2.twinx()  # mirror them
            ax4.set_yticks([16, 12, 8, 4, 0])
            app = np.array([-18, -16, -14, -12, -10]) + np.floor(5 * (np.log10(distance * 1e6) - 1))
            ax4.set_yticklabels(app.astype(int))
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
        if (not cnt == len(filts)) and (not cnt == 1):
            plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.set_zorder(1)
    ax2.set_xlabel('Time [days]', fontsize=48)
    return plt

#######

t = KNTable_read_samples_TSB('posterior_samples.dat', Nsamples=1000)

t = t.calc_tidal_lambda(remove_negative_lambda=True)
t = t.calc_compactness(fit=True)
t = t.calc_baryonic_mass(EOS=None, TOV=None, fit=True)
t = t.downsample(Nsamples=100)
tini = 0.1; tmax = 50.0; dt = 0.1; vmin = 0.02; th = 0.2; ph = 3.14; kappa = 1.0; eps = 1.58*(10**10); alp = 1.2; eth = 0.5; flgbct = 1; beta = 3.0; kappa_r = 1.0; slope_r = -1.2; theta_r = 0.0; Ye = 0.3
t['tini'] = tini; t['tmax'] = tmax; t['dt'] = dt; t['vmin'] = vmin; t['th'] = th; t['ph'] = ph; t['kappa'] = kappa; t['eps'] = eps; t['alp'] = alp; t['eth'] = eth; t['flgbct'] = flgbct; t['beta'] = beta; t['kappa_r'] = kappa_r; t['slope_r'] = slope_r; t['theta_r'] = theta_r; t['Ye'] = Ye

# Create dict of tables for the various models, calculating mass ejecta velocity of ejecta and the lightcurve from the model
models = ["DiUj2017","Me2017"]
model_tables = {}
for model in models:
    model_tables[model] = KNTable.model(model, t)
# Now we need to do some interpolation
for model in models:
    model_tables[model] = lightcurve_utils.calc_peak_mags(model_tables[model])
    model_tables[model] = lightcurve_utils.interpolate_mags_lbol(model_tables[model])

distance = 100 #Mpc
plot = plot_mag_panels_TSB(model_tables, distance=distance)
plot.savefig("kilonova_light_curves-example.png")












