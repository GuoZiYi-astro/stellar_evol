import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import time as t_module

class stellar_evol(object):
    '''
    Input parameters 
    dt : float 
        Duration of every timestep [yr]

        Default value: 4e6
    
    dm : float
        The resolution of stellar mass [Mo]

        Default value: 0.01
    tend : float
        Duraion of the whole evolution [yr]

        Default value: 13e9
    
    imf_type : string
        The type of imf

        Default value: 'kroupa'
    
    alpha1, alpha2, alpha3 : float
        For Kroupa IMF, this is the three index in three part

        Default value: -0.3, -1.3, -2.3

    imf_bdys : list
        Upper and lower mass limits of the initial mass functino (IMF) [Mo].

        Default value: [0.1, 100]
    
    m1, m2 : float
        For Kroupa IMF, the inner boundary of the three part [Mo].

        Default value: 0.08, 0.5
    
    sfh : list
        The shape of the list should be 2*n, the first column is time, and the second column is sfr [Mo].

        Default value: [[0,1],[13e9,1]]

    lifetime : list
        The lifetime of the stars with different mass. The shape of the list should be 2*n, the first column is stellar mass [Mo], and the second column is lifetime [yr]. In this code, we don't care the influence of metallicity.

        Default value: [[1, 13e9], [100, 3e6]]
    
    yield_list : list
        The yield list of the stars with different mass. The yield is all the yield of the star, do not contain the elements, and don't care the metallicity. The shape of the list should be 2*n, the first column is stellar mass [Mo], and the second column is all yield mass [Mo]

        Default value: [[1, 0.3],[100, 85]]
    '''
    def __init__(self, dt=3e6, dm=0.01, tend=13e9, imf_type='kroupa', alpha1=-0.3, alpha2=-1.3, \
        alpha3=-2.3, imf_bdys=[0.1,100], m1=0.5, m2=1, sfh=[[0,10],[13e9,10]], \
        lifetime=[[1,13e9],[40, 3e6], [100,3e6]], yield_list=[[1,0.3],[40, 20],[100,85]]):
        self.dt = dt
        self.dm = dm
        self.tend = tend
        self.imf_type = imf_type
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.imf_bdys = imf_bdys
        self.m1 = m1
        self.m2 = m2
        self.sfh = np.array(sfh)
        self.lifetime = np.array(lifetime)
        self.yield_list = np.array(yield_list)

        print('Star evolve stars')
        start_time = t_module.time()
        
        # self.A is the normalization parameter of imf number
        self.A = self.norm_imf_n()
        # self.B is the normalization parameter of imf mass
        self.B = self.norm_imf_m()

        # self.n_mass_bin is the number of the stellar mass bins
        self.n_mass_bin = self.get_int((self.imf_bdys[1]-self.imf_bdys[0])/self.dm)
        # self.n_time_step is the number of the time bins
        self.n_time_step = self.get_int(self.tend/self.dt)

        # self.star_num is the number evolution track of the stars in every stellar mass, only contain the alive stars.
        self.star_num = np.zeros((self.n_mass_bin+1, self.n_time_step))
        # self.star_mass is the mass evolution track of the stars in every stellar mass, only contain the alive stars.
        self.star_mass = np.zeros((self.n_mass_bin+1, self.n_time_step))
        # self.rem_mass is the mass of stellar remnant of the stars in every stellar mass, do not contain the alive stars.
        self.rem_mass = np.zeros((self.n_mass_bin+1, self.n_time_step))

        # self.star is the list of the stellar mass of all stars. This is the masses of our stellar mass bin.
        self.star = np.arange(self.imf_bdys[0], self.imf_bdys[1], (self.imf_bdys[1]-self.imf_bdys[0])/self.n_mass_bin)
        self.star = np.append(self.star, self.imf_bdys[-1])
        self.time = np.arange(0, tend, self.tend/self.n_time_step)

        # Interpolate the stellar lifetime to all stellar mass
        self.all_life = np.interp(self.star, self.lifetime[:,0], self.lifetime[:,1])
        # Then calculate the timesteps of all stellar masses
        all_life_step = np.zeros(len(self.all_life))
        for i_life in range(len(self.all_life)):
            all_life_step[i_life] = self.get_int(self.all_life[i_life]/self.dt)
        self.all_life_step = np.array(all_life_step)
        self.all_sfr = np.interp(self.time, self.sfh[:,0], self.sfh[:,1])

        # Interpolate the yield table to all stellar masses
        self.all_yield = np.interp(self.star, self.yield_list[:,0], self.yield_list[:,1]/self.yield_list[:,0])*self.star
        self.__test_valid()
        self.star_yield = np.zeros((self.n_mass_bin+1, self.n_time_step))
        
        for i, t in enumerate(self.time):
            self.__timestep(i)

        end_time = t_module.time()
        print('   Stars evolve completed - ' + str(round((end_time-start_time),2))+ ' s.')

    def __test_valid(self):
        for i_m,m in enumerate(self.star):
            if self.all_yield[i_m] > m:
                print('Something Wrong!!!!!')
                print(m)
                print(self.all_yield[i_m])
    
    def norm_imf_n(self):
        # normalize the IMF
        imf_num = integrate.quad(self.kroupa_imf, self.imf_bdys[0], self.imf_bdys[1])
        return 1/imf_num[0]
    
    def norm_imf_m(self):
        imf_m = integrate.quad(self.kroupa_imf_m, self.imf_bdys[0], self.imf_bdys[1], limit = 50)
        return 1/imf_m[0]

    def kroupa_imf(self, m):
        '''
        This is the definition of the kroupa imf number function. This function has not been normalized to number one.

        Input parameteres:
        m : the mass of stars, actually is the low limit of the mass bin.

        Return: 
        n : the number of this mass in this mass bin.
        '''
        a2 = self.m2**(self.alpha3-self.alpha2)
        a1 = a2*self.m1**(self.alpha2-self.alpha1)
        if m >= self.imf_bdys[0] and m < self.m1:
            return a1*m**self.alpha1
        elif m >= self.m1 and m < self.m2:
            return a2*m**self.alpha2
        elif m >= self.m2 and m < self.imf_bdys[1]:
            return m**self.alpha3
        else: 
            print('Out of boundary')
            return 0
    
    def kroupa_imf_m(self, m):
        '''
        This is the definition of the kroupa imf mass function. This function has not been normalized to number one.

        Input parameters:
        m : the mass of stars, actually is the low limit of the mass bin.

        Return:
        m : the mass of stars in this mass bin.
        '''
        a2 = self.m2**(self.alpha3-self.alpha2)
        a1 = a2*self.m1**(self.alpha2-self.alpha1)
        if m >= self.imf_bdys[0] and m < self.m1:
            return a1*m**(self.alpha1+1)
        elif m >= self.m1 and m < self.m2:
            return a2*m**(self.alpha2+1)
        elif m >= self.m2 and m < self.imf_bdys[1]:
            return m**(self.alpha3+1)
        else:
            print('Out of boundary')
            return 0
    
    def __timestep(self, i_t):
        '''
        In every timestep, calculate the stars form, and their dead time.
        '''
        sfr = self.all_sfr[i_t]*self.dt
        # B_t is the parameter of IMF in every timestep. Use this can achieve the sfr in this step
        B_t = sfr*self.B
        # i_mass is the real stellar mass of every stellar mass bin
        for j in range(self.n_mass_bin):
            i_num = B_t*integrate.quad(self.kroupa_imf,self.star[j], self.star[j+1])[0]
            i_mass = B_t*integrate.quad(self.kroupa_imf_m, self.star[j], self.star[j+1])[0]
            i_yield = i_num * self.all_yield[j]
            i_rem = i_mass - i_yield
            k = int(self.all_life_step[j])
            if k == 0:
                self.rem_mass[j,i_t+1:] += i_rem*np.ones(self.n_time_step-i_t-1)
                self.star_yield[j,i_t+1:] += i_yield*np.ones(self.n_time_step-i_t-1)
            elif i_t+k < self.n_time_step:
                self.star_num[j,i_t+1:i_t+k+1] += i_num*np.ones(k)
                self.star_mass[j,i_t+1:i_t+k+1] += i_mass*np.ones(k)
                self.star_yield[j, i_t+k+1:] += i_yield*np.ones(self.n_time_step-i_t-k-1)
                self.rem_mass[j,i_t+k+1:] += i_rem*np.ones(self.n_time_step-i_t-k-1)
                # print('Timestep: ' + str(i_t))
                # print('Time: '+str(self.time[i_t])+ 'yr')
                # print('Mass ' + str(self.star[j]))
            else:
                self.star_num[j,i_t+1:] += i_num*np.ones(self.n_time_step-i_t-1)
                self.star_mass[j,i_t+1:] += i_mass*np.ones(self.n_time_step-i_t-1)
   
    def get_int(self, f):
        if isinstance(f, list):
            f_int = []
            for i_f in f:
                f_int.append(int(i_f))
                if i_f-f_int[-1] < 0:
                    f_int[-1] += -1
            f_int = np.array(f_int)
        else:
            f_int = int(f)
            if f-f_int < 0 :
                f_int = f_int-1
        return f_int
         
    def plot_alive_stellar_mass(self, return_y = False, **kwargs):
        alive_stellar_mass = np.zeros(self.n_time_step)
        for i_m in range(self.n_time_step):
            alive_stellar_mass[i_m] = sum(self.star_mass[:,i_m])
        if return_y:
            return alive_stellar_mass
        plt.plot(self.time, alive_stellar_mass, **kwargs)
    
    def plot_all_stellar_mass(self, return_y = False, **kwargs):
        all_stellar_mass = np.zeros(self.n_time_step)
        for i_t in range(self.n_time_step):
            all_stellar_mass[i_t] = sum(self.star_mass[:,i_t])+sum(self.rem_mass[:,i_t])
        if return_y:
            return all_stellar_mass
        plt.plot(self.time, all_stellar_mass, **kwargs)

    def plot_yield(self, return_y = False, **kwargs):
        all_yield = np.zeros(self.n_time_step)
        for i_t in range(self.n_time_step):
            all_yield[i_t] = sum(self.star_yield[:,i_t])
        if return_y:
            return all_yield
        plt.plot(self.time, all_yield, **kwargs)
    
    def plot_sfh(self, **kwargs):
        plt.plot(self.sfh[:,0], self.sfh[:,1],**kwargs)
    
    def plot_imf(self, **kwargs):
        mass = np.arange(self.imf_bdys[0], self.imf_bdys[1], self.dm)
        d_num = []
        for i_m in mass:
            d_num.append(self.kroupa_imf(i_m))
        plt.plot(mass, d_num, **kwargs)
        plt.yscale('log')
        plt.xscale('log')
    
    def plot_t_stellar_mass(self, t, **kwargs):
        i_t = int(t/self.dt)
        plt.plot(self.star, self.star_mass[:,i_t], **kwargs)
    
    def print_alive_star(self, t):
        i_t = int(t/self.dt)
        print('Alive stellar mass at '+str(self.time[i_t])+' yr is : '+str(sum(self.star_mass[:,i_t]))+' Msun.')
        return sum(self.star_mass[:,i_t])
    
    def print_inte_star(self, t):
        i_t = int(t/self.dt)
        print('Integrated stellar mass at '+str(self.time[i_t])+' yr is : '+str(sum(self.star_mass[:,i_t])+sum(self.rem_mass[:,i_t])+sum(self.star_yield[:,i_t]))+' Msun.')
        return sum(self.star_mass[:,i_t])+sum(self.rem_mass[:,i_t])+sum(self.star_yield[:,i_t])

