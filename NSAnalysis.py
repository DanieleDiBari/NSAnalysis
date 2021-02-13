import numpy as np 
import h5py
import copy
import scipy.constants

import lmfit

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

kB_microeV_K = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0] * 1e6             # micro-eV / K

class Data:
    def __init__(
            self, 
            filename='', 
            filetype='hdf', 
            title = 'undefined',
            energy_rescale = 1e3,
            read_logfile_only=False, 
            correct_bad_data=False, 
            fix_signal_to_noise_range=0, 
            rm_no_signal=True, 
            rm_negative_signal=True,
            get_logdata=True,
            verbose=True,
            vana_filename='',
            read_vana_verbose=True,
        ):
        '''

            Read and correct QENS data.

            Acepted files:
             - Mantid \'hdf\' file
             - LAMP \'ascii\'

        '''
        self.title = title

        self.fname = 'undefined'
        self.ftype = 'undefined'

        self.global_title = 'undefined'
        self.subtitle     = 'undefined'
        self.q_title = 'undefined'
        self.E_title = 'undefined'
        self.S_title = 'undefined'
        
        # Main DATA
        self.q  = np.empty(0, dtype='f')
        self.dq = np.empty(0, dtype='f')
        self.energy_rescale = energy_rescale
        self.E  = np.empty(0, dtype='f')
        self.dE = np.empty(0, dtype='f')
        self.S  = np.empty((0,0), dtype='f')
        self.dS = np.empty((0,0), dtype='f')
        # DATA Correction
        self.data_correction = dict(log='No data correction!', rm_points=np.array([], dtype='i'))
        self.data_correction['types'] = (
            'Signal/Noise',
            'No Signal',
            'No Error',
            'NaN Signal',
            'NaN Error',
            'Neg. Signal',
            'Neg. Error',
            'Inf. Signal',
        )
        # Additional DATA
        self.sample_temp = -1.0
        self.wavelength  = -1.0
        self.instrument  = 'undefined'
        # Resolution DATA
        self.vana_read  = False

        self.logdata = 'undefined'
        
        if filename != '':
            self.read_data(filename, title=title, type_of_file=filetype, read_log_only=read_logfile_only, energy_rescale=energy_rescale)

        if correct_bad_data:
            if self.S.shape[0] >= 0:
                self.correct_data(
                    signal_to_noise_range = fix_signal_to_noise_range, 
                    rm_no_signal = rm_no_signal, 
                    rm_negative_values = rm_negative_signal, 
                    verbose = False
                )
        if get_logdata:
            self.get_logdata(verbose=verbose)

        if vana_filename != '':
            self.read_vana(vana_filename, verbose=read_vana_verbose)

    def read_data(self, filename, title='undefined', type_of_file='hdf', read_log_only=False, energy_rescale=1):
        self.title = title

        if not type_of_file in ['ascii', 'hdf']:
            raise Exception('ERROR! Type of file \'{}\' unknown.'.format(type_of_file))
        
        self.ftype = type_of_file
        if type_of_file == 'ascii':
            self.fname = dict(log=filename)
            datalog_lines = []
            with open(filename, 'r') as fin:
                datalog_lines = fin.readlines()
            
            get_x = False
            x_counter = -1
            get_y = False
            y_counter = -1

            for line in datalog_lines:
                if 'X_SIZE:' in line:
                    q_size = int(line.split()[1])
                    self.q = np.full(q_size, np.nan, dtype='f')
                elif 'Y_SIZE:' in line:
                    e_size = int(line.split()[1])
                    self.E = np.full(e_size, np.nan, dtype='f')
                elif 'TITLES:' in line:
                    self.global_title = line[9:-1]
                elif '      X:' in line:
                    self.q_title = line[9:-1]
                elif '      Y:' in line:
                    self.E_title = line[9:-1]
                elif '      Z:' in line:
                    self.S_title = line[9:-1]
                elif 'Sample   temperature [K]' in line:
                    self.sample_temp = float(line.split()[6])
                elif 'Wavelength   (angstroms)' in line:
                    self.wavelength = float(line.split()[5])
                elif '(Instrument) name:' in line:
                    self.instrument = line.split()[3]
                elif ' X_COORDINATES:' in line:
                    get_x = True
                elif get_x:
                    if '--><pre>' in line:
                        get_x = False
                    elif x_counter == -1:
                        x_counter += 1
                    elif x_counter < q_size:
                        for val in line.split():
                            self.q[x_counter] = float(val)
                            x_counter += 1
                elif ' Y_COORDINATES:' in line:
                    get_y = True
                elif get_y:
                    if '--><pre>' in line:
                        get_y = False
                    elif y_counter == -1:
                        y_counter += 1
                    elif y_counter < e_size:
                        for val in line.split():
                            self.E[y_counter] = float(val)
                            y_counter += 1
            
            if not read_log_only:
                def read_data(data_array, fname):
                    data_lines = []
                    with open(fname, 'r') as fin:
                        data_lines = fin.readlines()

                    x_size = len(data_array[0])
                    x_idx = 0
                    y_idx = 0

                    for line in data_lines:
                        for value in line.split():
                            data_array[y_idx][x_idx] = float(value)
                            x_idx += 1
                        if x_idx == x_size:
                            y_idx += 1
                            x_idx = 0
                
                self.S = np.full((self.E.shape[0], self.q.shape[0]), np.nan, dtype='f')
                read_data(self.S, filename+'ascii')
                self.S = self.S.transpose(1,0)
                self.fname['data'] = filename+'ascii'

                self.dS = np.full((self.E.shape[0], self.q.shape[0]), np.nan, dtype='f')
                read_data(self.dS, filename+'ascii_e')
                self.dS = self.dS.transpose(1,0)
                self.fname['err'] = filename+'ascii_e'
            else:
                self.fname['data'] = 'undefined'
                self.fname['err']  = 'undefined'
        else:
            with h5py.File(filename, 'r') as ifile: 
                # q
                self.q  = np.array(ifile['mantid_workspace_1/workspace/axis2'])
                self.dq = np.mean(self.q[1:] - self.q[0:-1])
                # E
                ## Note: Mantid save the energy array with one element more than for 
                ##       the intensities -> keep the mean value between to near elements
                energy_edges = np.array(ifile['mantid_workspace_1/workspace/axis1'])
                self.E  = 0.5 * (energy_edges[1:] + energy_edges[:-1]) 
                self.dE = np.mean(self.E[1:] - self.E[0:-1])
                # S
                self.S  = np.array(ifile['mantid_workspace_1/workspace/values'])
                self.dS = np.array(ifile['mantid_workspace_1/workspace/errors'])
                # Temperature
                self.sample_temp = np.array([
                    ifile['mantid_workspace_1/logs/sample.temperature/value'][:].mean(), 
                    ifile['mantid_workspace_1/logs/sample.temperature/value'][:].std()
                    ])
                # Pressure
                self.sample_pres = np.array([
                    ifile['mantid_workspace_1/logs/sample.pressure/value'][:].mean(), 
                    ifile['mantid_workspace_1/logs/sample.pressure/value'][:].std()
                    ])
                # Numors
                self.instrument = ifile['mantid_workspace_1/logs/instrument.name/value'][0].decode('utf-8')
                # Wavelength
                self.wavelength = ifile['mantid_workspace_1/logs/wavelength/value'][0]
                # Titles
                self.global_title = ifile['mantid_workspace_1/logs/title/value'][0].decode('utf-8')
                self.subtitle     = ifile['mantid_workspace_1/logs/subtitle/value'][0].decode('utf-8')
                # Filename 
                self.fname = filename
        
        if energy_rescale != 1:
            self.E = energy_rescale * self.E
            self.dE = energy_rescale * self.dE
            self.energy_rescale = energy_rescale

        self.x_window = [None for i_q in range(self.q.shape[0])]
        self.xlim = [None for i_q in range(self.q.shape[0])]
        for i_q in range(self.q.shape[0]):
            self.x_window[i_q] = (0,0)
            self.xlim[i_q] = (self.E.min(),self.E.max())

    def read_vana(self, vana_filename, verbose=False):           
        self.vana_read  = True
        self.vana_filename = vana_filename
        self.vana_data = read_FitDATA(vana_filename)
        if verbose:
            print('Read Vana DATA from: \'{}\'\n'.format(vana_filename))

    def correct_data(self, 
        signal_to_noise_range=0, 
        rm_no_signal=True, 
        rm_no_errors=True, 
        rm_negative_signal=True, 
        rm_negative_errors=True, 
        rm_infinite_signal=True, 
        delete_values=True,
        verbose=True, 
        row_length=20
        ):
        
        def delete_points(ids, del_val=True):
            if del_val:
                self.S[i][ids]  = 0
            self.dS[i][ids] = np.inf

        nq = self.q.shape[0]
        rm_datapoint = dict()
        for tdc in self.data_correction['types']:    
            rm_datapoint[tdc] = np.zeros(nq, dtype='i')
        for i in range(nq):
            # Reasonable signal-to-noise range
            if signal_to_noise_range > 0:
                ids_snr = self.S[i] < signal_to_noise_range * self.S[i].max()
                delete_points(ids_snr)
                rm_datapoint['Signal/Noise'][i] = self.S[i][ids_snr].shape[0]
            elif signal_to_noise_range == -1:
                signal_to_noise_range = 5e-4
                ids_snr = self.S[i] < signal_to_noise_range * self.S[i].max()
                delete_points(ids_snr)
                rm_datapoint['Signal/Noise'][i] = self.S[i][ids_snr].shape[0]
            # No signal
            elif rm_no_signal:
                ids_nosig = self.S[i] == 0
                delete_points(ids_nosig)
                rm_datapoint['No Signal'][i] = self.dS[i][ids_nosig].shape[0]
            # No errors
            if rm_no_errors:
                ids_noerr = self.dS[i] == 0
                delete_points(ids_noerr, del_val=delete_values)
                rm_datapoint['No Error'][i] = self.dS[i][ids_noerr].shape[0]
            # Negative signal
            if rm_negative_signal:
                ids_neg = self.S[i] < 0
                delete_points(ids_neg)
                rm_datapoint['Neg. Signal'][i] = self.S[i][ids_neg].shape[0]
            # Negative errors
            if rm_negative_errors:
                ids_negerr = self.dS[i] < 0
                delete_points(ids_negerr, del_val=delete_values)
                rm_datapoint['Neg. Error'][i] = self.dS[i][ids_negerr].shape[0]
            # Infinite values
            if rm_infinite_signal:
                ids_inf = np.isinf(self.S[i])
                delete_points(ids_inf)
                rm_datapoint['Inf. Signal'][i] = self.S[i][ids_inf].shape[0]
            # Nan values
            ids_nan = np.isnan(self.S[i])
            delete_points(ids_nan)
            rm_datapoint['NaN Signal'][i] = self.S[i][ids_nan].shape[0]
            # Nan errors
            ids_enan = np.isnan(self.dS[i])
            delete_points(ids_enan, del_val=delete_values)
            rm_datapoint['NaN Error'][i] = self.dS[i][ids_enan].shape[0]

        ntables = (nq-1) // row_length + 1
        rm_counter = np.zeros(nq, dtype='i')
        out = ''
        for i in range(ntables):
            qrange = [i * row_length, (i+1) * row_length]
            qrange[1] = nq if qrange[1] > nq else qrange[1]
            nq_row = qrange[1] - qrange[0]
            out += '\n  {:15}  |  '.format('Data Corr.')
            out += (nq_row * 'q_{:02}  ').format(*np.arange(qrange[0],qrange[1]))
            out += '\n'
            len_row = 22 + nq_row * 6
            out += '—' * len_row + '\n'
            for tdc in rm_datapoint:
                row  = '  {:15}  :  '.format(tdc)
                row += (nq_row * '{:4}  ').format(*rm_datapoint[tdc][qrange[0]:qrange[1]])
                out += row + '\n'
                rm_counter += rm_datapoint[tdc]
            out += '—' * len_row + '\n'
            out += '  {:15}  |  '.format('TOTALs')
            out += (nq_row * '{:4}  ').format(*rm_counter[qrange[0]:qrange[1]])
            out += '\n'
        if verbose:
            print(out[1:])
        self.data_correction['log'] = out 
        self.data_correction['rm_points'] = rm_counter
        
    def get_logdata(self, verbose=True):
        if self.ftype == 'ascii':
            if type(self.fname) == type(dict()):
                slash = len(self.fname['log']) - self.fname['log'][::-1].find('/')
                fname = self.fname['log'][slash:]
                fpath = self.fname['log'][:slash-1]
                logdata = ''
                logdata += 'Log file name == {}\n'.format(self.fname['log'])
                if self.fname['data'] != 'undefined':
                    logdata += 'Data file name == {}\n'.format(self.fname['data'])
                if self.fname['err'] != 'undefined':
                    logdata += 'Data_err file name == {}\n'.format(self.fname['err'])
                logdata += 'q.size == {}\n'.format(self.q.shape[0])
                logdata += 'E.size == {}\n'.format(self.E.shape[0])
                logdata += 'E (rescale)  == {}\n'.format(self.energy_rescale)
                logdata += 'S(q,E).size  == {}x{}\n'.format(self.S.shape[0], self.S.shape[1])
                logdata += 'global_title == {}\n'.format(self.global_title)
                logdata += 'x_title == {}\n'.format(self.q_title)
                logdata += 'y_title == {}\n'.format(self.E_title)
                logdata += 'z_title == {}\n'.format(self.S_title)
                logdata += 'sample_temp == {} +- {}\n'.format(self.sample_temp)  
                logdata += 'wavelength   == {}\n'.format(self.wavelength)
                logdata += 'instrument   == {}\n'.format(self.instrument)
            else:
                raise Exception('ERROR! No file read.')   
        else:
            if self.fname != 'undefined':
                slash = len(self.fname) - self.fname[::-1].find('/')
                fname = self.fname[slash:]
                fpath = self.fname[:slash-1]
                logdata = '\n'
                logdata += 'File name    == {}\n'.format(fname)
                logdata += 'File path    == {}\n'.format(fpath)
                logdata += 'q.size       == {}\n'.format(self.q.shape[0])
                logdata += 'E.size       == {}\n'.format(self.E.shape[0])
                logdata += 'E (rescale)  == {}\n'.format(self.energy_rescale)
                logdata += 'S(q,E).size  == {}x{}\n'.format(self.S.shape[0], self.S.shape[1])
                logdata += 'global_title == {}\n'.format(self.global_title)
                logdata += 'file_title   == {}\n'.format(self.subtitle)
                logdata += 'sample_temp  == ({} +- {}) K\n'.format(self.sample_temp[0], self.sample_temp[1]) 
                logdata += 'sample_pres  == ({} +- {}) Kbar\n'.format(self.sample_pres[0], self.sample_pres[1]) 
                logdata += 'wavelength   == {} Å\n'.format(self.wavelength)
                logdata += 'instrument   == {}\n'.format(self.instrument)
            else:
                raise Exception('ERROR! No file read.')   
        
        self.logdata = logdata
        
        if self.data_correction['log'] != 'No data correction!':
            logdata += self.data_correction['log']
        
        if verbose:
            print(logdata)

        return logdata
    
    def reduce_data(self, elim=[], qlim=[], rebin_avg=1, e_window=[]):
        if rebin_avg > 1:
            self.E = rebin(self.E, bin_avg=rebin_avg)
            temp_S  = np.empty(self.S.shape,  dtype='f') 
            temp_dS = np.empty(self.dS.shape, dtype='f') 
            for i in range(self.q.shape[0]):
                temp_S[i]  = rebin(self.S[i], bin_avg=rebin_avg)
                temp_dS[i] = rebin(self.dS[i], bin_avg=rebin_avg, error_prop=True)
            self.S  = temp_S
            self.dS = temp_dS

        if elim != [] or e_window != [] or qlim != []:
            # SET: Energies
            ### Excluded Energy Window - Get limits
            if len(e_window) == 1:
                ewmax = ewmin = np.abs(e_window[0])
            elif len(e_window) == 2:
                if e_window[0] > e_window[1]:
                    ewmax = e_window[0]
                    ewmin = e_window[1]
                elif e_window[1] > e_window[0]:
                    ewmax = e_window[1]
                    ewmin = e_window[0]
                else: 
                    ewmax = ewmin = np.abs(e_window[0])
                    print('WARNING! The energy window limits (e_window == [{},{}]) set for the data reduction are equals.\nData reducted simmetrically around zero: E <= -|e_window[0]|  or  E >= +|e_window[0]|'.format(ewmax, ewmax))
            elif len(e_window) > 2:
                raise Exception('ERROR! Setting energy limits for data reduction: input list too long')
            ### Maximal Energy Range - Get limits
            if len(elim) == 1:
                emax = emin = np.abs(elim[0])
            elif len(elim) == 2:
                if elim[0] > elim[1]:
                    emax = elim[0]
                    emin = elim[1]
                elif elim[1] > elim[0]:
                    emax = elim[1]
                    emin = elim[0]
                else: 
                    emax = emin = np.abs(elim[0])
                    print('WARNING! The energy limits (elim == [{},{}]) set for the data reduction are equals.\nData reducted simmetrically around zero: -|elim[0]| < E < +|elim[0]|'.format(emax, emax))
            elif len(e_window) > 2:
                raise Exception('ERROR! Setting energy limits for data reduction: input list too long')
            # Get Energy Indexes
            if elim == [] and e_window == []:
                eids = np.full(self.E.shape[0], True)
                self.e_window = (0.0, 0.0)
                self.elim = (self.E.min(), self.E.max())
            elif elim == []:
                eids =  (self.E <= ewmin) | (self.E >= ewmax)
                self.e_window = (ewmin, ewmax)
                self.elim = (self.E.min(), self.E.max())
            elif e_window == []:
                eids =  (self.E >= emin)  & (self.E <= emax)
                self.e_window = (0.0, 0.0)
                self.elim = (emin, emax)
            else:
                eids = ((self.E <= ewmin) | (self.E >= ewmax)) & ((self.E >= emin) & (self.E <= emax))
                self.e_window = (ewmin, ewmax)
                self.elim = (emin, emax)

            # SET: q values
            if qlim == []:
                qids = np.full(self.q.shape[0], True)
            elif len(qlim) == 2:
                if qlim[0] != qlim[1]:
                    if qlim[0] > qlim[1]:
                        qmax = qlim[0]
                        qmin = qlim[1]
                    elif qlim[1] > qlim[0]:
                        qmax = qlim[1]
                        qmin = qlim[0]
                    qids = (self.q >= qmin) & (self.q <= qmax)
                else: 
                    raise Exception('ERROR! The q limits (qlim == [{},{}]) set for the data reduction are equals.'.format(emax, emax))
            else:
                raise Exception('ERROR! Setting q limits for data reduction: input list with wrong dimension.')
        
            ## Storing removed data
            not_eids = np.invert(eids)
            self.rm_E  = self.E[not_eids]  
            self.rm_S  = self.S[qids][:,not_eids]
            self.rm_dS = self.dS[qids][:,not_eids]
            ## Reducing Data
            self.E  = self.E[eids]                    # <-    emin < E < emax   &   ewmin < E < ewmax       
            self.q  = self.q[qids]                    # <-    qmin < q < qmax               
            self.S  = self.S[qids][:,eids]
            self.dS = self.dS[qids][:,eids]

            if len(qlim) == 2:
                self.x_window = [None for i_q in range(self.q.shape[0])]
                self.xlim = [None for i_q in range(self.q.shape[0])]
                for i_q in range(self.q.shape[0]):
                    self.x_window[i_q] = (0,0)
                    self.xlim[i_q] = (self.E.min(),self.E.max())

        elif rebin_avg == 1:
            print('WARNING! Data reduction: Nothing done.')

    def get_xy(self, i_q, elim=[], e_window=[], rebin_avg=1):
        x_val = copy.deepcopy(self.E)
        y_val = copy.deepcopy(self.S[i_q])
        y_err = copy.deepcopy(self.dS[i_q])

        if rebin_avg > 1:
            x_val = rebin(x_val, bin_avg=rebin_avg)
            y_val = rebin(y_val,  bin_avg=rebin_avg)
            y_err = rebin(y_err, bin_avg=rebin_avg)

        if elim != [] or e_window != []:
            # SET: Energies
            ### Excluded Energy Window - Get limits
            if len(e_window) == 1:
                ewmax = ewmin = np.abs(e_window[0])
            elif len(e_window) == 2:
                if e_window[0] > e_window[1]:
                    ewmax = e_window[0]
                    ewmin = e_window[1]
                elif e_window[1] > e_window[0]:
                    ewmax = e_window[1]
                    ewmin = e_window[0]
                else: 
                    ewmax = ewmin = np.abs(e_window[0])
                    print('WARNING! The energy window limits (e_window == [{},{}]) set for the data reduction are equals.\nData reducted simmetrically around zero: E <= -|e_window[0]|  or  E >= +|e_window[0]|'.format(ewmax, ewmax))
            elif len(e_window) > 2:
                raise Exception('ERROR! Setting energy limits for data reduction: input list too long')
            ### Maximal Energy Range - Get limits
            if len(elim) == 1:
                emax = emin = np.abs(elim[0])
            elif len(elim) == 2:
                if elim[0] > elim[1]:
                    emax = elim[0]
                    emin = elim[1]
                elif elim[1] > elim[0]:
                    emax = elim[1]
                    emin = elim[0]
                else: 
                    emax = emin = np.abs(elim[0])
                    print('WARNING! The energy limits (elim == [{},{}]) set for the data reduction are equals.\nData reducted simmetrically around zero: -|elim[0]| < E < +|elim[0]|'.format(emax, emax))
            elif len(e_window) > 2:
                raise Exception('ERROR! Setting energy limits for data reduction: input list too long')
            # Get Energy Indexes
            if elim == [] and e_window == []:
                eids = np.full(x_val, True)
                self.x_window[i_q] = (0.0, 0.0)
            elif elim == []:
                eids =  (x_val <= ewmin) | (x_val >= ewmax)
                self.x_window[i_q] = (ewmin, ewmax)
            elif e_window == []:
                eids =  (x_val >= emin)  & (x_val <= emax)
                self.x_window[i_q] = (0.0, 0.0)
            else:
                eids = ((x_val <= ewmin) | (x_val >= ewmax)) & ((x_val >= emin) & (x_val <= emax))
                self.x_window[i_q] = (ewmin, ewmax)

            ## Reducing Data
            x_val = x_val[eids]                    # <-    emin < E < emax   &   ewmin < E < ewmax       
            y_val = y_val[eids]
            y_err = y_err[eids]
        
        ## Remove Bad Data
        ids = y_err != np.inf
        x_val = x_val[ids]      
        y_val = y_val[ids]
        y_err = y_err[ids]

        self.xlim[i_q] = (x_val.min(), x_val.max())

        return x_val, np.array([y_val, y_err])

    def qfit(
        self, fit_function, par_hints, 
        elim=[], e_window=[], rebin_avg=1, 
        par_hints_qvec=dict(), 
        detailed_balance_factor=True, 
        fit_method='leastsq', 
        sigma_vana_pname='sigma_vana', 
        center_pname='center', 
        extern_center_vana_pname='center', 
        extern_sigma_vana_pname='sigma',
        use_vana_for_center=True, 
        contiguos_par_hints=False, 
        starting_qid=0, 
        log_title='', log_filename='', 
        data_title='', data_subtitle='', data_filename='', 
        odir='', 
        verbose=False
        ):

        print('START QFit')
        if not self.vana_read and use_vana_for_center:
            raise Exception('ERROR! \"use_vana_for_center\" set True, but no Vana Parameters readed.')

        self.qfit_method  = fit_method 
        self.qfit_model  = [None for i_q in range(self.q.shape[0])]
        self.qfit_params = [None for i_q in range(self.q.shape[0])]
        self.qfit_result = [None for i_q in range(self.q.shape[0])]
        self.qfit_end_params = dict()
        for par in par_hints:
            self.qfit_end_params[par]   = np.full((2, self.q.shape[0]), np.inf)
        self.qfit_end_params['red_chi_squared'] = np.full((2, self.q.shape[0]), np.inf)
        self.qfit_end_params['en_resolution']   = np.full((2, self.q.shape[0]), np.inf)

        if self.vana_read:
            self.qfit_end_params[sigma_vana_pname]  = np.full((2, self.q.shape[0]), np.inf)

            q_vana = self.vana_data['q']
            q_vana_id = []
            for qi in self.q:
                for j_q, qj in enumerate(q_vana):
                    if '{:4.2f}'.format(qi) == '{:4.2f}'.format(qj):
                    #if np.round(qi,5) == np.round(q,5):
                        q_vana_id.append(j_q)
                        break
            q_vana_id = np.array(q_vana_id)
            center_vana = self.vana_data[extern_center_vana_pname]
            sigma_vana  = self.vana_data[extern_sigma_vana_pname]

        if data_title == '':
            if self.ftype == 'ascii':
                data_title = self.title + ' {} K - QFIT Results'.format(self.sample_temp)
            else:
                data_title = self.title + ' ( {} +- {} ) K - QFIT Results'.format(*self.sample_temp)
        if data_subtitle == '':
            data_subtitle = 'Function: ' + fit_function.__name__
        result_DATA = data_title + '\n' + data_subtitle + '\n'
        result_DATA += '{:>15}  {:>10}  '.format('q', 'red_chi^2')
        for par in par_hints:
            result_DATA += '{:>27}  {:>26}  '.format(par, par+'_stderr')
        if self.vana_read:
            result_DATA += '{:>27}  {:>26}  '.format(sigma_vana_pname, sigma_vana_pname+'_stderr')
        result_DATA += '\n'
        if data_filename == '':
            data_filename = self.title + '_{}K.QFitDATA.txt'.format(int(np.round(self.sample_temp if self.ftype == 'ascii' else self.sample_temp[0])))

        if log_title == '':
            if self.ftype == 'ascii':
                data_title = self.title + ' {} K - QFIT Results log'.format(self.sample_temp)
            else:
                data_title = self.title + ' ( {} +- {} ) K - QFIT Results log'.format(*self.sample_temp)
        log =  '\n|' + '-' * 78 + '|\n|' + '-' * 12 + '|' + ' ' * 8
        log += log_title
        log += ' ' * 8 + '|' + '-' * 12 + '|\n|' + '-' * 78 + '|\n'
        log += 'INPUT FILE INFO:\n'
        log += self.logdata + '\n|' + '-' * 78 + '|\n'
        self.qfit_result_LOG = log
        if log_filename == '':
            log_filename = self.title + '_{}K.QFitLOG.txt'.format(int(np.round(self.sample_temp if self.ftype == 'ascii' else self.sample_temp[0])))
        
        if starting_qid != 0:
            qid_stdrange = np.arange(0, self.q.shape[0], 1, dtype='i')
            qid_range = np.zeros(self.q.shape[0], dtype='i')
            qid_shift = self.q.shape[0] - starting_qid
            qid_range[:qid_shift] = copy.copy(qid_stdrange[starting_qid:])
            qid_range[qid_shift:] = copy.copy(qid_stdrange[starting_qid-1::-1])
        else:
            qid_range = np.arange(0, self.q.shape[0], 1, dtype='i')

        for i_q  in qid_range:
            qi = self.q[i_q]
            x, y = self.get_xy(i_q, elim=elim, e_window=e_window, rebin_avg=rebin_avg)
            
            if detailed_balance_factor:
                self.detailed_balance_factor = detailed_balance_factor
                y = y / self.db_factor(x)
            self.qfit_model[i_q] = lmfit.Model(fit_function)
            
            for par in par_hints:
                hints = par_hints[par]
                if par == center_pname and use_vana_for_center:
                    hints['value'] = center_vana[0, q_vana_id[i_q]]
                elif par in par_hints_qvec:
                    hints['value'] = par_hints_qvec[par]['value'][i_q]
                elif contiguos_par_hints:
                    if i_q > starting_qid:
                        hints['value'] = self.qfit_end_params[par][0, i_q-1]
                    if i_q < starting_qid:
                        hints['value'] = self.qfit_end_params[par][0, i_q+1]
                self.qfit_model[i_q].set_param_hint(par, **hints)

            if self.vana_read:
                self.qfit_model[i_q].set_param_hint(sigma_vana_pname, value=sigma_vana[0,q_vana_id[i_q]], vary=False)
            self.qfit_params[i_q] = self.qfit_model[i_q].make_params()

            self.qfit_result[i_q] = self.qfit_model[i_q].fit(
                y[0], 
                self.qfit_params[i_q], 
                x=x, 
                weights=1.0/y[1], 
                method=self.qfit_method
            )

            for par in self.qfit_params[i_q].keys():
                self.qfit_end_params[par][0,i_q] = self.qfit_result[i_q].params[par].value
                if type(self.qfit_result[i_q].params[par].stderr) == type(self.qfit_result[i_q].params[par].value):
                    self.qfit_end_params[par][1,i_q] = self.qfit_result[i_q].params[par].stderr
                else:
                    if par == center_pname and use_vana_for_center:
                        self.qfit_end_params[par][1,i_q] = center_vana[1, q_vana_id[i_q]]
                    elif par == sigma_vana_pname and use_vana_for_center:
                        self.qfit_end_params[par][1,i_q] = sigma_vana[1, q_vana_id[i_q]]
                    elif par_hints[par]['vary']:
                        self.qfit_end_params[par][1,i_q] = np.inf
                        print('   WARNING! q_id = {:2d} , q_val = {:4.2f} : NO ERROR for PARAMETER \'{}\''.format(i_q,qi,par))
                    else:
                        self.qfit_end_params[par][1,i_q] = 0
            #red_chisqrt = self.qfit_result[i_q].chisqr / (self.qfit_result[i_q].ndata - self.data_correction['rm_points'][i_q] - self.qfit_result[i_q].nvarys)
            self.qfit_end_params['red_chi_squared'][0,i_q] = self.qfit_result[i_q].redchi
            if self.vana_read:
                self.qfit_end_params['en_resolution'][:,i_q]   = self.qfit_end_params[sigma_vana_pname][:,i_q] * (8*np.log(2))**0.5

            result_DATA += '{:15.13f}  {:10.4e}  '.format(qi, self.qfit_result[i_q].redchi)
            for par in par_hints:
                if self.qfit_end_params[par][1,i_q] != np.inf:
                    result_DATA += '{:+22.20e}  {:22.20e}  '.format(
                        self.qfit_end_params[par][0,i_q], 
                        self.qfit_end_params[par][1,i_q]
                    )
                else:
                    result_DATA += '{:+22.20e}  {:26}  '.format(
                        self.qfit_end_params[par][0,i_q], 
                        self.qfit_end_params[par][1,i_q]
                    )
            if self.vana_read:
                if self.qfit_end_params[sigma_vana_pname][1,i_q] != np.inf:
                    result_DATA += '{:+22.20e}  {:22.20e}'.format(
                        self.qfit_end_params[sigma_vana_pname][0,i_q], 
                        self.qfit_end_params[sigma_vana_pname][1,i_q]
                    )
                else:
                    result_DATA += '{:+22.20e}  {:26}  '.format(
                        self.qfit_end_params[sigma_vana_pname][0,i_q], 
                        self.qfit_end_params[sigma_vana_pname][1,i_q]
                    )
            result_DATA += '\n'
                
            temp_result_LOG = '#'*35 + ' q = {:4.2f} '.format(qi) + '#'*35 + '\n' + str(self.qfit_result[i_q].fit_report(min_correl=0.25)) + '\n' + '#'*80 + '\n'
            if verbose:
                print(temp_result_LOG)
            self.qfit_result_LOG += '\n' + temp_result_LOG
        self.qfit_result_DATA = result_DATA

        print('END QFit (Avg. RedChi^2 == {})\n'.format(self.qfit_end_params['red_chi_squared'][0].mean()))

        with open(odir+log_filename, 'w') as fout:
            fout.write(self.qfit_result_LOG)
            print('Saved: ' + odir + log_filename)

        with open(odir+data_filename, 'w') as fout:
            fout.write(self.qfit_result_DATA)
            print('Saved: ' + odir + data_filename)

    def qfit_plot(
        self,
        fig_title = '', fig_title_y = 0.918, fig_title_fs = 24, fig_title_fw = 600,
        dpi = 100,
        n_col = 6,  dim_col = 12,
        n_row = 3,  dim_row = 10,
        legend_anchor = (0.009, 0.985), legend_loc='upper left',
        qbox_pos=dict(x=0.975, y=1.03), qbox_align=dict(h='right', v='center'),
        fit_neval = 1000,
        normalize = True, ylim_upper='common', ylim_lower='common',
        elim=[], e_window=[], rebin_avg=1, 
        xlabel = 'Energy [$\mu eV$]',
        ylabel = 'Intensity [arb. unit]',
        sigma_vana_pname='sigma_vana', center_pname='center', 
        fit_function_components='none'
        ):
        if n_row == -1 or (n_row*n_col) < self.q.shape[0]:
            n_row = self.q.shape[0] // n_col
            if self.q.shape[0] % n_col > 0:
                n_row += 1
        
        if fig_title == '':
            if self.ftype == 'ascii':
                fig_title = self.title + ' {:6.2f}K - QFit'.format(self.sample_temp)
            else:
                fig_title = self.title + ' ($\\bf{{{:6.2f}\pm{:4.2f}}}$)K - QFit'.format(*self.sample_temp)
        fig = plt.figure(figsize=(n_col * dim_col, n_row * dim_row), dpi=dpi)
        fig.suptitle(fig_title, y=fig_title_y, size=fig_title_fs, fontweight=fig_title_fw)

        main_gs = gridspec.GridSpec(n_row, n_col, figure=fig)#, hspace=0.09, wspace=0.25)
        axs = []
        for i in range(n_row):
            for j in range(n_col):
                sub_gs = main_gs[i,j].subgridspec(2, 1, height_ratios=[4,1], hspace=0.05)
                axs.append([fig.add_subplot(sub_gs[0]), fig.add_subplot(sub_gs[1])])
    
        self.qfit_elastic_summedintensity = np.zeros((2, self.q.shape[0]))
        self.qfit_summedintensity = np.zeros((2, self.q.shape[0]))
        self.qfit_ylims = dict(min=np.full((2, self.q.shape[0]), np.inf), max=np.full((2, self.q.shape[0]), -np.inf))
        qfit_residues_ylims = dict(min=np.full((2, self.q.shape[0]), np.inf), max=np.full((2, self.q.shape[0]), -np.inf))
        
        if elim == []:
            set_elim = True
        else:
            set_elim = False
        for i_q, qi in enumerate(self.q):
            if set_elim:
                elim = self.xlim[i_q]

            x, y = self.get_xy(i_q, elim=elim)
            x_fitdata, y_fitdata = self.get_xy(i_q, elim=self.xlim[i_q])
            x_fit = np.linspace(x[0], x[-1], num=fit_neval)

            if self.detailed_balance_factor:
                db_factor_fit = self.db_factor(x_fit)
                db_factor_fitdata = self.db_factor(x_fitdata)
            else:
                db_factor_fit = 1
                db_factor_fitdata = 1

            m  = self.qfit_result[i_q].eval(x=x_fit) * db_factor_fit
            dm = self.qfit_result[i_q].eval_uncertainty(x=x_fit) * db_factor_fit
            
            self.qfit_ylims['max'][0, i_q] = m.max()
            ids_max = (m == m.max())
            if np.count_nonzero(ids_max) == 1:
                self.qfit_ylims['max'][1, i_q] = dm[ids_max]
            else:
                self.qfit_ylims['max'][1, i_q] = dm[ids_max].mean()
            self.qfit_ylims['min'][0, i_q] = m.min()
            ids_min = (m == m.min())
            if np.count_nonzero(ids_min) == 1:
                self.qfit_ylims['min'][1, i_q] = dm[ids_min]
            else:
                self.qfit_ylims['min'][1, i_q] = dm[ids_min].mean()

            elastic_erange = (x_fit >= -self.qfit_end_params['en_resolution'][0, i_q]) & (x_fit <= self.qfit_end_params['en_resolution'][0, i_q])
            self.qfit_elastic_summedintensity[0, i_q] = m[elastic_erange].mean()
            self.qfit_elastic_summedintensity[1, i_q] = np.sqrt((dm[elastic_erange]**2).mean())
            self.qfit_summedintensity[0, i_q] = m.sum()
            self.qfit_summedintensity[1, i_q] = np.sqrt((dm**2).sum())

            if normalize:
                norm = 1 / self.qfit_ylims['max'][0, i_q]
                m = norm * m
                dm = norm * dm
            else:
                norm = 1
            if self.vana_read:
                gauss = lmfit.models.GaussianModel().func
                en_res = gauss(x_fit, amplitude=1, center=self.qfit_end_params[center_pname][0][i_q], sigma=self.qfit_end_params[sigma_vana_pname][0][i_q])
                en_res = en_res / en_res.max()

            ax = axs[i_q][0]
            bx = axs[i_q][1]

            alpha_errorbar = 0.3

            markers, caps, bars = ax.errorbar(x,    norm * y[0],    yerr=norm * y[1],    fmt='o', color='gray', label='Data')
            [bar.set_alpha(alpha_errorbar) for bar in bars]
            [cap.set_alpha(alpha_errorbar) for cap in caps]
            ax.axvspan(self.x_window[i_q][0], self.x_window[i_q][1], color='k', alpha=0.1)
            ax.axvspan(-np.inf, self.xlim[i_q][0], color='k', alpha=0.1)
            ax.axvspan(self.xlim[i_q][1], +np.inf, color='k', alpha=0.1)
            ax.fill_between(x_fit, m+dm,m-dm, color='r', alpha=0.25)
            f = ax.plot(x_fit, m, 'r', label='Fit')
            df = ax.fill(np.NaN, np.NaN, color='r', alpha=0.25)
            ax.fill_between(x_fit, m+dm,m-dm, color='r', alpha=0.25)  

            if type(fit_function_components) != type('none'):
                parms = dict()
                excluded_pars = ['red_chi_squared','en_resolution']
                for p in self.qfit_end_params:
                    if p not in excluded_pars:
                        parms[p] = self.qfit_end_params[p][0][i_q]
                components = fit_function_components(x=x_fit, **parms)
                for cname in components:
                    y_fit = norm * components[cname] * db_factor_fit
                    ax.plot(x_fit, y_fit, '--', label=cname)
                    '''
                    if len(y_fit) == 1:
                        ax.plot((x_fit[0], x_fit[-1]), (y_fit, y_fit), '--', label=cname)
                    else:
                        ax.plot(x_fit, y_fit, '--', label=cname)
                    '''

            if self.vana_read:
                ax.plot(x_fit, en_res, ':', color='gray', label='Resolution')
            
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
            ax.text(qbox_pos['x'], qbox_pos['y'], '$\\bf{{q \:=\: {:5.3f} \, \AA^{{-1}}}}$'.format(qi), size=12, horizontalalignment=qbox_align['h'], verticalalignment=qbox_align['v'], transform=ax.transAxes, bbox=bbox_props)
            ax.set_xticks([])

            bx.plot(elim, (0,0), color='k', lw=1, ls='-.')
            m_f = self.qfit_result[i_q].best_fit * db_factor_fitdata
            dm_f = self.qfit_result[i_q].eval_uncertainty(x=x_fitdata) * db_factor_fitdata
            residues = np.array([
                m_f - y_fitdata[0],
                np.sqrt(dm_f**2 + y_fitdata[1]**2)
                ])
            f_b = bx.plot(x,   1000*residues[0], color='b')
            bx.fill_between(x, 1000*(residues[0]+residues[1]), 1000*(residues[0]-residues[1]), color='magenta', alpha=0.45)
            df_b = bx.fill(np.NaN, np.NaN, color='magenta', alpha=0.45)

            ax.set(ylabel=ylabel, xlim=elim)
            bx.set(xlabel=xlabel, xlim=elim)

            ax.legend(bbox_to_anchor=legend_anchor, loc=legend_loc)

            if type(ylim_lower) == type(''):
                qfit_residues_ylims['max'][0, i_q] = residues[0].max()
                ids_max = (residues[0] == residues[0].max())
                if np.count_nonzero(ids_max) == 1:
                    qfit_residues_ylims['max'][1, i_q] = residues[1][ids_max]
                else:
                    qfit_residues_ylims['max'][1, i_q] = residues[1][ids_max].mean()
                qfit_residues_ylims['min'][0, i_q] = residues[0].min()
                ids_min = (residues[0] == residues[0].min())
                if np.count_nonzero(ids_min) == 1:
                    qfit_residues_ylims['min'][1, i_q] = residues[1][ids_min]
                else:
                    qfit_residues_ylims['min'][1, i_q] = residues[1][ids_min].mean()

        for i_q, qi in enumerate(self.q):
            ax = axs[i_q][0]
            bx = axs[i_q][1]

            if not normalize:
                if type(ylim_upper) == type(''):
                    if ylim_upper == 'common':
                        dy = 0.075 * np.max(self.qfit_ylims['max'][0] + self.qfit_ylims['max'][1])
                        ax.set(ylim=(self.qfit_ylims['min'][0].min() - dy, self.qfit_ylims['max'][0].max() + dy))
                    elif ylim_upper == 'q-dependent':
                        dy = 0.075 * (self.qfit_ylims['max'][0] + self.qfit_ylims['max'][1])
                        ax.set(ylim=(self.qfit_ylims['min'][0, i_q] - dy, self.qfit_ylims['max'][0, i_q] + dy))
                    else:
                        raise Exception('ERROR! Unknown value for limits of the uppers graphs: \"{}\".\nAllowed settings are:\nAUTOMATIC SETTING\n - \"common\": set the same ylims for all the upper graphs\n - \"q-dependent\": set the different ylims for each upper graphs\nMANUAL SETTING\n - (<ymin>, <ymax>): set the ylims as <ymin> and <ymax> for all the upper graphs'.format(ylim_upper))
                elif type(ylim_upper) == type((1,1)):
                    ax.set(ylim=ylim_upper)
                else:
                    raise Exception('ERROR! Unknown value for limits of the uppers graphs: \"{}\".\nAllowed settings are:\nAUTOMATIC SETTING\n - \"common\": set the same ylims for all the upper graphs\n - \"q-dependent\": set the different ylims for each upper graphs\nMANUAL SETTING\n - (<ymin>, <ymax>): set the ylims as <ymin> and <ymax> for all the upper graphs'.format(ylim_upper))
        
            if type(ylim_lower) == type(''):
                if ylim_lower == 'common':
                    ymax = 1000 * np.max(qfit_residues_ylims['max'][0] + qfit_residues_ylims['max'][1])
                    ymin = 1000 * np.min(qfit_residues_ylims['min'][0] - qfit_residues_ylims['min'][1])
                    if ymax < -ymin:
                        dy = - 1.075 * ymin
                    else:
                        dy = 1.075 * ymax
                    bx.set(ylim=(-dy, dy))
                elif ylim_lower == 'q-dependent':
                    ymax = 1000 * (qfit_residues_ylims['max'][0, i_q] + qfit_residues_ylims['max'][1, i_q])
                    ymin = 1000 * (qfit_residues_ylims['min'][0, i_q] - qfit_residues_ylims['min'][1, i_q])
                    if ymax < -ymin:
                        dy = - 1.075 * ymin
                    else:
                        dy =   1.075 * ymax
                    bx.set(ylim=(-dy, dy))
                else:
                    raise Exception('ERROR! Unknown value for limits of the lower graphs: \"{}\".\nAllowed settings are:\nAUTOMATIC SETTING\n - \"common\": set the same ylims for all the lower graphs\n - \"q-dependent\": set the different ylims for each lower graphs\nMANUAL SETTING\n - (<ymin>, <ymax>): set the ylims as <ymin> and <ymax> for all the lower graphs'.format(ylim_lower))
            elif type(ylim_lower) == type((1,1)):
                bx.set(ylim=ylim_lower)
            else:
                raise Exception('ERROR! Unknown value for limits of the lower graphs: \"{}\".\nAllowed settings are:\nAUTOMATIC SETTING\n - \"common\": set the same ylims for all the lower graphs\n - \"q-dependent\": set the different ylims for each lower graphs\nMANUAL SETTING\n - (<ymin>, <ymax>): set the ylims as <ymin> and <ymax> for all the lower graphs'.format(ylim_lower))
        

        self.qfit_fig = fig
        self.qfit_axs = axs

    def simultaneus_fit(
        self, fit_dataset_function, shared_phints, not_shared_phints, 
        not_shared_phints_qvec=dict(), fit_method='leastsq', 
        sigma_vana_pname='sigma_vana', center_pname='center', 
        extern_center_vana_pname='center', extern_sigma_vana_pname='sigma',
        use_vana_for_center=True, 
        log_title='', log_filename='', 
        data_title='', data_subtitle='', data_filename='', 
        odir='', verbose=False, detailed_balance_factor=True,
        with_error=True):
        
        print('START SFit')
        self.sfit_dataset_function = fit_dataset_function
        self.sfit_method = fit_method

        def objective(params, x, q, data, std=None):
            """ Calculate total residual for fits of Gaussians to several data sets """
            ndata, _ = data.shape
            resid = 0.0*data[:]

            # make residual per data set
            if std is None:
                for i in range(ndata):
                    resid[i, :] = fit_dataset_function(params, i, q[i], x) - data[i, :]
            else:
                for i in range(ndata):
                    resid[i, :] = (fit_dataset_function(params, i, q[i], x) - data[i, :]) / std[i, :]
        
            # now flatten this to a 1D array, as minimize() needs
            return resid.flatten()

        self.sfit_end_params = dict(shared={}, not_shared={})
        for par in shared_phints:
            self.sfit_end_params['shared'][par] = np.full(2, np.inf)
        self.sfit_end_params['shared']['red_chi_squared'] = np.full(2, np.inf)
        for i_q in range(self.q.shape[0]):
            for par in not_shared_phints:
                par = par + '_' + str(i_q+1)
                self.sfit_end_params['not_shared'][par] = np.full(2, np.inf)
            par = sigma_vana_pname + '_' + str(i_q+1)
            self.sfit_end_params['not_shared'][par] = np.full(2, np.inf)
            par = 'en_resolution' + '_' + str(i_q+1)
            self.sfit_end_params['not_shared'][par] = np.full(2, np.inf)

        q_vana = self.vana_data['q']
        q_vana_id = []
        for qi in self.q:
            for j_q, qj in enumerate(q_vana):
                if '{:4.2f}'.format(qi) == '{:4.2f}'.format(qj):
                    q_vana_id.append(j_q)
                    break
        q_vana_id = np.array(q_vana_id)
        center_vana = self.vana_data[extern_center_vana_pname]
        sigma_vana  = self.vana_data[extern_sigma_vana_pname]

        if data_title == '':
            if self.ftype == 'ascii':
                data_title = self.title + ' {} K - SFIT Results'.format(self.sample_temp)
            else:
                data_title = self.title + ' ( {} +- {} ) K - SFIT Results'.format(*self.sample_temp)
        if data_subtitle == '':
            data_subtitle = 'Function: ' + fit_dataset_function.__name__
        result_DATA = data_title + '\n' + data_subtitle + '\nNOT SHARED Parameters:\n'
        result_DATA += '{:>15}  '.format('q')
        for par in not_shared_phints:
            result_DATA += '{:>27}  {:>26}  '.format(par, par+'_stderr')
        result_DATA += '{:>27}  {:>26}  '.format(sigma_vana_pname, sigma_vana_pname+'_stderr')
        result_DATA += '\n'
        if data_filename == '':
            data_filename = self.title + '_{}K.SFitDATA.txt'.format(int(np.round(self.sample_temp if self.ftype == 'ascii' else self.sample_temp[0])))

        if log_title == '':
            if self.ftype == 'ascii':
                data_title = self.title + ' {} K - SFIT Results log'.format(self.sample_temp)
            else:
                data_title = self.title + ' ( {} +- {} ) K - SFIT Results log'.format(*self.sample_temp)
        log =  '\n|' + '-' * 78 + '|\n|' + '-' * 12 + '|' + ' ' * 8
        log += log_title
        log += ' ' * 8 + '|' + '-' * 12 + '|\n|' + '-' * 78 + '|\n'
        log += 'INPUT FILE INFO:\n'
        log += self.logdata + '\n|' + '-' * 78 + '|\n'
        self.sfit_result_LOG = log
        if log_filename == '':
            log_filename = self.title + '_{}K.SFitLOG.txt'.format(int(np.round(self.sample_temp if self.ftype == 'ascii' else self.sample_temp[0])))
        
        self.sfit_params = lmfit.Parameters()
        # Shared (among q values)
        for par in shared_phints:
            hints = shared_phints[par]
            self.sfit_params.add(par, **hints)
        for i_q in range(self.q.shape[0]):
            for par in not_shared_phints:
                hints = not_shared_phints[par]
                if par == center_pname and use_vana_for_center:
                    hints['value'] = center_vana[0, q_vana_id[i_q]]
                elif par in not_shared_phints_qvec:
                    hints['value'] = not_shared_phints_qvec[par]['value'][i_q]
                par = par + '_' + str(i_q + 1)
                self.sfit_params.add(par, **hints)
            self.sfit_params.add(sigma_vana_pname + '_' + str(i_q + 1), value=sigma_vana[0, q_vana_id[i_q]], vary=False)
        
        norm = 1
        if detailed_balance_factor:
            self.detailed_balance_factor = detailed_balance_factor
            norm = 1 / self.db_factor(self.E)
        if with_error:
            self.sfit_result = lmfit.minimize(fcn=objective, params=self.sfit_params, kws=dict(x=self.E, q=self.q, data=norm*self.S, std=norm*self.dS), method=self.sfit_method)
        else:
            self.sfit_result = lmfit.minimize(fcn=objective, params=self.sfit_params, kws=dict(x=self.E, q=self.q, data=norm*self.S, std=None), method=self.sfit_method)
        if verbose:
            lmfit.report_fit(self.sfit_result.params)
        #self.sfit_result_LOG += '\n' + lmfit.report_fit(self.sfit_result.params)

        for i_q in range(self.q.shape[0]):
            qi = self.q[i_q]
            for par in not_shared_phints:
                pname  = par + '_' + str(i_q+1)
                value  = self.sfit_result.params.get(pname).value
                stderr = self.sfit_result.params.get(pname).stderr
                self.sfit_end_params['not_shared'][pname][0] = value
                if type(stderr) == type(value):
                    self.sfit_end_params['not_shared'][pname][1] = stderr
                else:
                    if par == center_pname:
                        self.sfit_end_params['not_shared'][pname][1] = center_vana[1, q_vana_id[i_q]]
                    elif not_shared_phints[par]['vary']:
                        self.sfit_end_params['not_shared'][pname][1] = np.inf
                        print('   WARNING! q_id = {:2d} , q_val = {:4.2f} : NO ERROR for PARAMETER \'{}\''.format(i_q, qi, pname))
                    else:
                        self.sfit_end_params['not_shared'][pname][1] = 0
            self.sfit_end_params['not_shared'][sigma_vana_pname+'_'+str(i_q+1)][0] = sigma_vana[0, q_vana_id[i_q]]
            self.sfit_end_params['not_shared'][sigma_vana_pname+'_'+str(i_q+1)][1] = sigma_vana[1, q_vana_id[i_q]]
            self.sfit_end_params['not_shared']['en_resolution_'+str(i_q+1)] = self.sfit_end_params['not_shared'][sigma_vana_pname+'_'+str(i_q+1)] * (8*np.log(2))**0.5
        
        red_chisqrt = self.sfit_result.chisqr / (self.sfit_result.ndata - self.data_correction['rm_points'][i_q] - self.sfit_result.nvarys)    
        self.sfit_end_params['shared']['red_chi_squared'][0] = red_chisqrt
        for par in shared_phints:
            value  = self.sfit_result.params.get(par).value
            stderr = self.sfit_result.params.get(par).stderr
            self.sfit_end_params['shared'][par][0] = value
            if type(stderr) == type(value):
                self.sfit_end_params['shared'][par][1] = stderr
            else:
                if shared_phints[par]['vary']:
                    self.sfit_end_params['shared'][par][1] = np.inf
                    print('   WARNING! NO ERROR for SHARED PARAMETER \'{}\''.format(par))
                else:
                    self.sfit_end_params['shared'][par][1] = 0
            
        
        for i_q in range(self.q.shape[0]):
            qi = self.q[i_q]
            result_DATA += '{:15.13f}  '.format(qi)
            for par in not_shared_phints:
                pname  = par + '_' + str(i_q+1)
                ns_par = self.sfit_end_params['not_shared'][pname]
                if ns_par[1] != np.inf:
                    result_DATA += '{:+22.20e}  {:22.20e}  '.format(ns_par[0], ns_par[1])
                else:
                    result_DATA += '{:+22.20e}  {:26}  '.format(ns_par[0], ns_par[1])
            pname  = sigma_vana_pname + '_' + str(i_q+1)
            ns_par = self.sfit_end_params['not_shared'][pname]
            if ns_par[1] != np.inf:
                result_DATA += '{:+22.20e}  {:22.20e}  '.format(ns_par[0], ns_par[1])
            else:
                result_DATA += '{:+22.20e}  {:26}  '.format(ns_par[0], ns_par[1])
            result_DATA += '\n'

        result_DATA += 'SHARED Parameters:\n'
        result_DATA += '{:>10}  '.format('red_chi^2')
        for par in shared_phints:
            result_DATA += '{:>27}  {:>26}  '.format(par, par+'_stderr')
        result_DATA += '\n'

        result_DATA += '{:10.4e}  '.format(self.sfit_end_params['shared']['red_chi_squared'][0])    
        for par in shared_phints:
            s_par = self.sfit_end_params['shared'][par]
            if s_par[1] != np.inf:
                result_DATA += '{:+22.20e}  {:22.20e}  '.format(s_par[0], s_par[1])
            else:
                result_DATA += '{:+22.20e}  {:26}  '.format(s_par[0], s_par[1])
        result_DATA += '\n'

        self.sfit_result_DATA = result_DATA

        print('END SFit (RedChi^2 == {})\n'.format(self.sfit_end_params['shared']['red_chi_squared'][0]))

        with open(odir+log_filename, 'w') as fout:
            fout.write(self.sfit_result_LOG)
            print('Saved: ' + odir + log_filename)

        with open(odir+data_filename, 'w') as fout:
            fout.write(self.sfit_result_DATA)
            print('Saved: ' + odir + data_filename)
             
    def sfit_plot(
        self,
        fig_title = '', fig_title_y = 0.918, fig_title_fs = 24, fig_title_fw = 600,
        dpi = 100,
        n_col = 6,  dim_col = 12,
        n_row = 3,  dim_row = 10,
        legend_anchor = (0.009, 0.985), legend_loc='upper left',
        qbox_pos=dict(x=0.975, y=1.03), qbox_align=dict(h='right', v='center'),
        fit_neval = 1000,
        normalize = True,
        xlabel = 'Energy [$\mu eV$]',
        ylabel = 'Intensity [arb. unit]',
        sigma_vana_pname='sigma_vana', center_pname='center', 
        fit_function_components='none'
        ):
        if n_row == -1 or (n_row*n_col) < self.q.shape[0]:
            n_row = self.q.shape[0] // n_col
            if self.q.shape[0] % n_col > 1:
                n_row += 1

        if fig_title == '':
            if self.ftype == 'ascii':
                fig_title = self.title + ' {:6.2f}K - QFit'.format(self.sample_temp)
            else:
                fig_title = self.title + ' ($\\bf{{{:6.2f}\pm{:4.2f}}}$)K - QFit'.format(*self.sample_temp)
        fig = plt.figure(figsize=(n_col * dim_col, n_row * dim_row), dpi=dpi)
        fig.suptitle(fig_title, y=fig_title_y, size=fig_title_fs, fontweight=fig_title_fw)

        main_gs = gridspec.GridSpec(n_row, n_col, figure=fig)#, hspace=0.09, wspace=0.25)
        axs = []
        for i in range(n_row):
            for j in range(n_col):
                sub_gs = main_gs[i,j].subgridspec(2, 1, height_ratios=[4,1], hspace=0.05)
                axs.append([fig.add_subplot(sub_gs[0]), fig.add_subplot(sub_gs[1])])

        x_fit = np.linspace(self.E[0], self.E[-1], num=fit_neval)

        self.qfit_elastic_summedintensity = np.zeros((2,self.q.shape[0]))
        self.qfit_summedintensity = np.zeros((2,self.q.shape[0]))
        for i_q, qi in enumerate(self.q):
            
            if self.detailed_balance_factor:
                db_factor_fit = self.db_factor(x_fit)
                db_factor_fitdata = self.db_factor(self.E)
            else:
                db_factor_fit = 1
                db_factor_fitdata = 1

            m  = self.sfit_dataset_function(self.sfit_result.params, i_q, qi, x_fit) * db_factor_fit
            dm = np.zeros(m.shape[0])

            elastic_erange = (x_fit >= -self.sfit_end_params['not_shared']['en_resolution_'+str(i_q+1)][0]) & (x_fit <= self.sfit_end_params['not_shared']['en_resolution_'+str(i_q+1)][1])
            self.qfit_elastic_summedintensity[0, i_q] = m[elastic_erange].mean()
            self.qfit_elastic_summedintensity[1, i_q] = np.sqrt((dm[elastic_erange]**2).mean())
            self.qfit_summedintensity[0, i_q] = m.sum()
            self.qfit_summedintensity[1, i_q] = np.sqrt((dm**2).sum())

            if normalize:
                norm = 1 / m.max() 
                m = norm * m
                dm = norm * dm
            else:
                norm = 1

            gauss = lmfit.models.GaussianModel().func
            en_res = gauss(x_fit, amplitude=1, 
                center=self.sfit_end_params['not_shared'][center_pname+'_'+str(i_q+1)][0], 
                sigma=self.sfit_end_params['not_shared'][sigma_vana_pname+'_'+str(i_q+1)][0])
            en_res = en_res / en_res.max()

            ax = axs[i_q][0]
            bx = axs[i_q][1]

            alpha_errorbar = 0.3
            markers, caps, bars = ax.errorbar(self.E,    norm * self.S[i_q],    yerr=norm * self.dS[i_q],    fmt='o', color='gray', label='Data')
            [bar.set_alpha(alpha_errorbar) for bar in bars]
            [cap.set_alpha(alpha_errorbar) for cap in caps]
            markers, caps, bars = ax.errorbar(self.rm_E, norm * self.rm_S[i_q], yerr=norm * self.rm_dS[i_q], fmt='o', color='gray')
            [bar.set_alpha(alpha_errorbar) for bar in bars]
            [cap.set_alpha(alpha_errorbar) for cap in caps]
            ax.axvspan(self.e_window[0], self.e_window[1], color='k', alpha=0.1)
            ax.fill_between(x_fit, m+dm,m-dm, color='r', alpha=0.25)
            f = ax.plot(x_fit, m, 'r', label='Fit')
            df = ax.fill(np.NaN, np.NaN, color='r', alpha=0.25)
            ax.fill_between(x_fit, m+dm,m-dm, color='r', alpha=0.25)  

            if type(fit_function_components) != type('none'):
                parms = dict()
                excluded_pars = ['red_chi_squared','en_resolution']
                for p in self.sfit_end_params:
                    if p not in excluded_pars:
                        parms[p] = self.sfit_end_params[p][0]
                components = fit_function_components(x=x_fit, **parms)
                for cname in components:
                    y_fit = norm * components[cname] * db_factor_fit
                    ax.plot(x_fit, y_fit, '--', label=cname)
                    '''
                    if len(y_fit) == 1:
                        ax.plot((x_fit[0], x_fit[-1]), (y_fit, y_fit), '--', label=cname)
                    else:
                        ax.plot(x_fit, y_fit, '--', label=cname)
                    '''

            ax.plot(x_fit, en_res, ':', color='gray', label='Resolution')

            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
            ax.text(qbox_pos['x'], qbox_pos['y'], '$\\bf{{q \:=\: {:5.3f} \, \AA^{{-1}}}}$'.format(qi), size=12, horizontalalignment=qbox_align['h'], verticalalignment=qbox_align['v'], transform=ax.transAxes, bbox=bbox_props)
            ax.set_xticks([])

            bx.plot(self.elim, (0,0), color='k', lw=1, ls='-.')
            m_f = self.sfit_dataset_function(self.sfit_result.params, i_q, qi, self.E) * db_factor_fitdata
            dm_f = 0#self.sfit_result[i_q].eval_uncertainty(x=self.E)
            diff = m_f - self.S[i_q]
            diff_err = np.sqrt(dm_f**2 + self.dS[i_q]**2)
            f_b = bx.plot(self.E,   1000*diff, color='b')
            bx.fill_between(self.E, 1000*(diff+diff_err), 1000*(diff-diff_err), color='magenta', alpha=0.45)
            df_b = bx.fill(np.NaN, np.NaN, color='magenta', alpha=0.45)

            ax.set(ylabel=ylabel, xlim=self.elim)
            bx.set(xlabel=xlabel, xlim=self.elim)

            ax.legend(bbox_to_anchor=legend_anchor, loc=legend_loc)
        
        self.sfit_fig = fig
        self.sfit_axs = axs

    def db_factor(self, x):
        if self.ftype == 'ascii':
            return np.e**(- x / (2 * kB_microeV_K * self.sample_temp))
        else:
            return np.e**(- x / (2 * kB_microeV_K * self.sample_temp[0]))

def read_FitDATA(fname, n_headerlines=3):
    with open(fname, 'r') as fin:
        txt = fin.readlines()
    
    nh = n_headerlines
    n = len(txt) - nh

    vdim = dict() 
    for i, k in enumerate(txt[nh-1].split()): 
        if 'stderr' not in k: 
            k_err = k + '_stderr' 
            if k_err in vdim: 
                vdim[k][0] += 1 
                vdim[k][1][0] = i 
            else: 
                vdim[k] = [1, [i, -1]] 
        else: 
            k_noerr = k[:-7] 
            if k_noerr in vdim: 
                vdim[k_noerr][0] += 1 
                vdim[k_noerr][1][1] = i 
            else: 
                vdim[k_noerr] = [1, [-1, i]] 

    data = dict() 
    for k in vdim: 
        if vdim[k][0] == 1: 
            data[k] = np.full(len(txt)-nh, np.NaN) 
        else: 
            data[k] = np.full((vdim[k][0], len(txt)-nh), np.NaN) 

    for i in range(n):
        for k in vdim: 
            line = txt[nh+i].split()
            if vdim[k][0] == 1: 
                data[k][i] = line[vdim[k][1][0]]
            else:
                data[k][0,i] = line[vdim[k][1][0]]
                data[k][1,i] = line[vdim[k][1][1]]

    return data

def rebin(x, bin_avg=2, verbose=True, error_prop=False):
    if bin_avg == 1:
        return x
    else:
        npoint_old = x.shape[0]
        off_set = npoint_old % bin_avg
        npoint_new = npoint_old//bin_avg
        if not error_prop:
            new_x = x[off_set:].reshape(npoint_new, bin_avg).mean(1)
        else:
            new_x = np.sqrt((x[off_set:].reshape(npoint_new, bin_avg)**2).sum(1))/bin_avg
        return new_x

def get_q(theta, hw, gamma0):
    '''
    
    FROM:
        theta   =  angle between the incident and the scattered neutron [degree]
        hw      =  energy change experienced by the sample [meV] 
        gamma0  =  wave lenght of the incident neutron 
    
    COMPUTE:
        q       =  difference between the incident and scattered wave vector of the neutron [Å]
    
    '''

    h = scipy.constants.physical_constants['Planck constant'] 
    e = scipy.constants.physical_constants['elementary charge']
    m = scipy.constants.physical_constants['neutron mass']
    
    w_meter = gamma0 * 1e-10
    theta_rad = theta * np.pi / 180
    k0 = 2 * np.pi / gamma0
    e0 = 0.5 * h[0]**2 / (m[0] * w_meter**2)
    e0_meV = 1e3 * e0 / e[0]            
    r = hw / e0_meV
    q = k0 * np.sqrt(2 - np.outer(r, np.ones(theta_rad.shape[0])) - 2 * np.outer(np.sqrt(1 - r), np.cos(theta_rad)))
    q.sort(axis=1)
    return q[0]
