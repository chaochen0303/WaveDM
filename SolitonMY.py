import numpy as np
import numexpr as ne
import numba as nb
import sys, os
import math
import matplotlib.pyplot as plt
import scipy.integrate as si
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl
import matplotlib.animation
import matplotlib.colors as mcolors
#import pyfftw
from scipy.interpolate import RegularGridInterpolator


#########-----units------########################


hbar = 1.0545718e-34  # m^2 kg/s
parsec = 3.0857e16  # m
light_year = 9.4607e15  # m
solar_mass = 1.989e30  # kg
axion_mass = 1e-22 * 1.783e-36  # kg
G = 6.67e-11  # N m^2 kg^-2
omega_m0 = 0.31
H_0 = 67.7 * 1e3 / (parsec * 1e6)  # s^-1

length_unit = (8 * np.pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25 #m
time_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** -0.5# second
mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G) #kg

#-------convert to dimensionless untis-----------------

def convertMY(value, unit, type, axion_mass, v0, lamb): 
#axion_mass:1e-22 eV; v0:100km/s; lamb: rescaling
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm'):
            converted = value/(1e3*parsec) /lamb * 5.2 * axion_mass/1e-22 * v0/100
        elif (unit == 'km'):
            converted = value/parsec /lamb * 5.2 * axion_mass/1e-22 * v0/100
        elif (unit == 'pc'):
            converted = value/1e3 /lamb * 5.2 * axion_mass/1e-22 * v0/100
        elif (unit == 'kpc'):
            converted = value /lamb * 5.2 * axion_mass/1e-22 * v0/100
        elif (unit == 'Mpc'):
            converted = value/1e-3 /lamb * 5.2 * axion_mass/1e-22 * v0/100
        elif (unit == 'ly'):
            converted = value/(1e3 * parsec / light_year)  /lamb * 5.2 * axion_mass/1e-22 * v0/100
        else:
            raise NameError('Unsupported length unit used')


    elif (type == 'm'):
        if (unit == ''):
            converted = value
        elif (unit == 'kg'):
            converted =  value/(1e8 * solar_mass) * 4.5 * lamb * axion_mass/1e-22 / v0/100
        elif (unit == 'solar_masses'):
            converted = value/1e8 * 4.5 * lamb * axion_mass/1e-22 / v0/100 
        else:
            raise NameError('Unsupported mass unit used')


    elif (type == 't'):
        if (unit == ''):
            converted = value
        elif (unit == 's'):
            converted =  value /(13.8*1e9*3600 * 24 * 365) / lamb**2 * (7.35e3) * axion_mass/1e-22 * (v0/100)**2
        elif (unit == 'yr'):
            converted =  value /(13.8*1e9) / lamb**2 * (7.35e3) * axion_mass/1e-22 * (v0/100)**2
        elif (unit == 'kyr'):
            converted = value /(13.8*1e6) / lamb**2 * (7.35e3) * axion_mass/1e-22 * (v0/100)**2
        elif (unit == 'Myr'):
            converted = value /(13.8*1e3) / lamb**2 * (7.35e3) * axion_mass/1e-22 * (v0/100)**2
        elif (unit == 'Gyr'):
            converted = value / 13.8 / lamb**2 * (7.35e3) * axion_mass/1e-22 * (v0/100)**2    
        else:
            raise NameError('Unsupported mass unit used')        


    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'km/s'):
            converted = value/100 /v0/100 * lamb
        else:
            raise NameError('Unsupported speed unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted


def convert(value, unit, type, lamb):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value/lamb
        elif (unit == 'm'):
            converted = value/lamb / length_unit
        elif (unit == 'km'):
            converted = value/lamb * 1e3 / length_unit
        elif (unit == 'pc'):
            converted = value/lamb * parsec / length_unit
        elif (unit == 'kpc'):
            converted = value/lamb * 1e3 * parsec / length_unit
        elif (unit == 'Mpc'):
            converted = value/lamb * 1e6 * parsec / length_unit
        elif (unit == 'ly'):
            converted = value/lamb * light_year / length_unit
        else:
            raise NameError('Unsupported length unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value* lamb
        elif (unit == 'kg'):
            converted = value* lamb / mass_unit
        elif (unit == 'solar_masses'):
            converted = value* lamb * solar_mass / mass_unit
        elif (unit == 'M_solar_masses'):
            converted = value* lamb * solar_mass * 1e6 / mass_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 't'):
        if (unit == ''):
            converted = value/lamb**2
        elif (unit == 's'):
            converted = value/lamb**2 / time_unit
        elif (unit == 'yr'):
            converted = value/lamb**2 * 60 * 60 * 24 * 365 / time_unit
        elif (unit == 'kyr'):
            converted = value/lamb**2 * 60 * 60 * 24 * 365 * 1e3 / time_unit
        elif (unit == 'Myr'):
            converted = value/lamb**2 * 60 * 60 * 24 * 365 * 1e6 / time_unit
        elif (unit == 'Gyr'):
            converted = value/lamb**2 * 60 * 60 * 24 * 365 * 1e9 / time_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value*lamb
        elif (unit == 'm/s'):
            converted = value*lamb * time_unit / length_unit
        elif (unit == 'km/s'):
            converted = value*lamb * 1e3 * time_unit / length_unit
        elif (unit == 'km/h'):
            converted = value*lamb * 1e3 / (60 * 60) * time_unit / length_unit
        else:
            raise NameError('Unsupported speed unit used')

    elif (type == 'angular'):
        if (unit == ''):
            converted = value*lamb
        elif (unit == 'normal'): #msun.mpc.km/s
            converted = value*lamb / (axion_mass/1e-23/ 1.783e-36)**(-5/2) / 1.1 / 1e+6
        else: 
            raise NameError('Unsupported speed unit used')

    elif (type == 'g'): 
        if (unit == ''):
            converted = value/lamb**2
        elif (unit == 'GeV-2'):
            converted =  3.5 * 1e+5 * (value/1e-21) * (axion_mass/1e-23/1.783e-36)**(-1) /lamb**2
     
        else:
            raise NameError('Unsupported mass unit used')


    else:
        raise TypeError('Unsupported conversion type')

    return converted

####################### FUNCTION TO CONVERT FROM DIMENSIONLESS UNITS TO DESIRED UNITS

def convert_back(value, unit, type,lamb):
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value*lamb
        elif (unit == 'm'):
            converted = value*lamb * length_unit
        elif (unit == 'km'):
            converted = value*lamb / 1e3 * length_unit
        elif (unit == 'pc'):
            converted = value*lamb / parsec * length_unit
        elif (unit == 'kpc'):
            converted = value*lamb / (1e3 * parsec) * length_unit
        elif (unit == 'Mpc'):
            converted = value*lamb / (1e6 * parsec) * length_unit
        elif (unit == 'ly'):
            converted = value*lamb / light_year * length_unit
        else:
            raise NameError('Unsupported length unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value/lamb
        elif (unit == 'kg'):
            converted = value/lamb * mass_unit
        elif (unit == 'solar_masses'):
            converted = value/lamb / solar_mass * mass_unit
        elif (unit == 'M_solar_masses'):
            converted = value/lamb / (solar_mass * 1e6) * mass_unit
        else:
            raise NameError('Unsupported mass unit used')

    elif (type == 't'):
        if (unit == ''):
            converted = value*lamb**2
        elif (unit == 's'):
            converted = value*lamb**2 * time_unit
        elif (unit == 'yr'):
            converted = value*lamb**2 / (60 * 60 * 24 * 365) * time_unit
        elif (unit == 'kyr'):
            converted = value*lamb**2 / (60 * 60 * 24 * 365 * 1e3) * time_unit
        elif (unit == 'Myr'):
            converted = value*lamb**2 / (60 * 60 * 24 * 365 * 1e6) * time_unit
        elif (unit == 'Gyr'):
            converted = value*lamb**2 / (60 * 60 * 24 * 365 * 1e9) * time_unit
        else:
            raise NameError('Unsupported time unit used')

    elif (type == 'v'):
        if (unit == ''):
            converted = value/lamb
        elif (unit == 'm/s'):
            converted = value/lamb / time_unit * length_unit
        elif (unit == 'km/s'):
            converted = value/lamb / (1e3) / time_unit * length_unit
        elif (unit == 'km/h'):
            converted = value/lamb / (1e3) * (60 * 60) / time_unit * length_unit
        else:
            raise NameError('Unsupported speed unit used')

    elif (type == 'angular'):
        if (unit == ''):
            converted = value/lamb
        elif (unit == 'normal'): #msun.mpc.km/s
            converted = value/lamb * (axion_mass/1e-23/ 1.783e-36)**(-5/2) * 1.1 * 1e+6
        else: 
            raise NameError('Unsupported speed unit used')

    elif (type == 'g'):
        if (unit == ''):
            converted = value * lamb**2
        elif (unit == 'GeV-2'):
            converted = value*1e-21 /3.5 / 1e+5 / (axion_mass/1e-23/1.783e-36)**(-1) * lamb**2
    else:
        raise TypeError('Unsupported conversion type')

    return converted

def coupling(fa):# fa in unit of GeV, g in unit of GeV^-2
    return - 1/ 8 / fa**2 #=g; <0
def coupling_back(g):
    return np.sqrt(-1/8/g) #=fa

##------covert from dimensionless units to desired units-------------

def convertMY_back(value, unit, type, axion_mass, v0, lamb):
    #axion_mass:1e-22 eV; v0:100km/s; lamb: rescaling
    converted = 0
    if (type == 'l'):
        if (unit == ''):
            converted = value
        elif (unit == 'm'):
            converted = value*(1e3*parsec) *lamb / 5.2 / axion_mass*1e-22 / v0*100
        elif (unit == 'km'):
            converted = value * parsec *lamb / 5.2 / axion_mass*1e-22 / v0*100
        elif (unit == 'pc'):
            converted = value*1e3 * lamb / 5.2 / axion_mass*1e-22 / v0* 100
        elif (unit == 'kpc'):
            converted = value * lamb / 5.2 / axion_mass * 1e-22 / v0 * 100
        elif (unit == 'Mpc'):
            converted = value*1e-3 *lamb / 5.2 / axion_mass*1e-22 / v0*100
        elif (unit == 'ly'):
            converted = value * (1e3 * parsec / light_year)  * lamb / 5.2 / axion_mass*1e-22 / v0*100
        else:
            raise NameError('Unsupported length unit used')

    elif (type == 'm'):
        if (unit == ''):
            converted = value
        elif (unit == 'kg'):
            converted = value*(1e8 * solar_mass) / 4.5 / lamb / axion_mass*1e-22 * v0*100
        elif (unit == 'solar_masses'):
            converted = value*1e8 / 4.5 / lamb / axion_mass*1e-22 * v0*100 
        else:
            raise NameError('Unsupported mass unit used')


    elif (type == 't'):
        if (unit == ''):
            converted = value
        elif (unit == 's'):
            converted = value * 13.8*1e9*3600 * 24 * 365 * lamb**2/(7.35e3)/axion_mass*1e-22 / (v0/100)**2
        elif (unit == 'yr'):
            converted = value * 13.8*1e9 * lamb**2/(7.35e3)/axion_mass*1e-22 / (v0/100)**2
        elif (unit == 'kyr'):
            converted = value * 13.8*1e6 * lamb**2/(7.35e3)/axion_mass*1e-22 / (v0/100)**2
        elif (unit == 'Myr'):
            converted = value * 13.8*1e3 * lamb**2/(7.35e3)/axion_mass*1e-22 / (v0/100)**2
        elif (unit == 'Gyr'):
            converted = value * 13.8 * lamb**2 / (7.35e3) / axion_mass*1e-22 / (v0/100)**2  
        else:
            raise NameError('Unsupported time unit used')



    elif (type == 'v'):
        if (unit == ''):
            converted = value
        elif (unit == 'km/s'):
            converted = value*100 *v0*100 / lamb
        else:
            raise NameError('Unsupported speed unit used')

    else:
        raise TypeError('Unsupported conversion type')

    return converted




######Runge-Kutta 4th order algorithm to solve for the initial soliton profile##########
def initial_profile(dr, max_radius, npy, dat, PLOT, g, BHmass):#BHmass=[BH1,BH2,...]

    rge = max_radius/dr #steps
    avg_mass = sum(BHmass) / len(BHmass)

    #equations needed to solve
    def g1(r, f, psi, fprime):# g: self-interaction strength; g < 0: attractive.
        return -(2/r)*fprime + 2*psi*f + 2 * g * f**3 - 2*(avg_mass / r)*f

    def g2(r, f, psiprime):
        return f**2 - (2/r)*psiprime

    # a=f, b=psi, c=f', d=psi'
    s = 0.
    diff = 1.
    optimised = False

    while optimised == False:

        ai = 1
        bi = s
        ci = 0
        di = 0

        la = []
        lb = []
        lc = []
        ld = []
        lr = []
        intlist = []

        la.append(ai)
        lb.append(bi)
        lc.append(ci)
        ld.append(di)
        lr.append(dr/100) # intial radius
        intlist.append(0.)

        # kn lists follow index a, b, c, d, i.e. k1[0] is k1a
        k1 = []
        k2 = []
        k3 = []
        k4 = []
        
        #4th-order RK
        for i in range(int(rge)):

            list1 = []
            list1.append(lc[i]*dr)
            list1.append(ld[i]*dr)
            list1.append(g1(lr[i],la[i],lb[i],lc[i])*dr)
            list1.append(g2(lr[i],la[i],ld[i])*dr)
            k1.append(list1)

            list2 = []
            list2.append((lc[i]+k1[i][2]/2)*dr)
            list2.append((ld[i]+k1[i][3]/2)*dr)
            list2.append(g1(lr[i]+dr/2,la[i]+k1[i][0]/2,lb[i]+k1[i][1]/2,lc[i]+k1[i][2]/2)*dr)
            list2.append(g2(lr[i]+dr/2,la[i]+k1[i][0]/2,ld[i]+k1[i][3]/2)*dr)
            k2.append(list2)

            list3 = []
            list3.append((lc[i]+k2[i][2]/2)*dr)
            list3.append((ld[i]+k2[i][3]/2)*dr)
            list3.append(g1(lr[i]+dr/2,la[i]+k2[i][0]/2,lb[i]+k2[i][1]/2,lc[i]+k2[i][2]/2)*dr)
            list3.append(g2(lr[i]+dr/2,la[i]+k2[i][0]/2,ld[i]+k2[i][3]/2)*dr)
            k3.append(list3)

            list4 = []
            list4.append((lc[i]+k3[i][2])*dr)
            list4.append((ld[i]+k3[i][3])*dr)
            list4.append(g1(lr[i]+dr,la[i]+k3[i][0],lb[i]+k3[i][1],lc[i]+k3[i][2])*dr)
            list4.append(g2(lr[i]+dr,la[i]+k3[i][0],ld[i]+k3[i][3])*dr)
            k4.append(list4)

            la.append(la[i]+(k1[i][0]+2*k2[i][0]+2*k3[i][0]+k4[i][0])/6)
            lb.append(lb[i]+(k1[i][1]+2*k2[i][1]+2*k3[i][1]+k4[i][1])/6)
            lc.append(lc[i]+(k1[i][2]+2*k2[i][2]+2*k3[i][2]+k4[i][2])/6)
            ld.append(ld[i]+(k1[i][3]+2*k2[i][3]+2*k3[i][3]+k4[i][3])/6)
            lr.append(lr[i]+dr)
            intlist.append((la[i]+(k1[i][0]+2*k2[i][0]+2*k3[i][0]+k4[i][0])/6)**2*(lr[i]+dr)**2) #f^2 (r+dr)^2
            
            #shoot method
            if (k1[i][0]+2*k2[i][0]+2*k3[i][0]+k4[i][0])/6>0:
                #print('{}{}'.format('Starting to diverge to +inf or oscillate at r = ', lr[i]))
                s = s-diff
                break

            if la[i]+(k1[i][0]+2*k2[i][0]+2*k3[i][0]+k4[i][0])/6<0:
                s = s + diff
                diff = diff/10
                s = s - diff
                break

            if i == int(rge)-1:
                optimised = True
                print('{}{}'.format('Successfully optimised for s = ', s))
                grad = (lb[i] - lb[i - 1]) / dr #psi'
                const = lr[i] ** 2 * grad #c
                beta = lb[i] + const / lr[i] #beta

    #Calculate full width at half maximum density:
    difflist = []
    for i in range(int(rge)):
        difflist.append(abs(la[i]**2 - 0.5)) #(f-1/2) for f(r=0) = 1/2
    fwhm = 2*lr[difflist.index(min(difflist))] # 2*rhalf, the width

    #Calculate the (dimensionless) total mass of the soliton: 
    mass = si.simps(intlist,lr)*4*np.pi

    print ('{}{}'.format('M_core = ', 0.237*mass))

    #Calculate the radius containing 90% of the mass
    partial = 0.
    for i in range(int(rge)):
        partial = partial + intlist[i]*4*np.pi*dr
        if partial >= 0.9*mass:
            r90 = lr[i]
            break

    #partial = 0.
    #for i in range(int(rge)):
    #    partial = partial + intlist[i]*4*np.pi*dr
    #    if lr[i] >= 0.5*1.38:
    #        print ('{}{}'.format('M_core = ', partial))
    #        break


    print ('{}{}'.format('mass is ', mass))
    print ('{}{}'.format('Full width at half maximum density is ', fwhm))
    print ('{}{}'.format('Beta is ', beta))
    print ('{}{}'.format('Radius at 90% mass is ', r90))

    #Save data
    psi_rd_ini = np.array(la) # intial soliton profile
    if (npy):
        file_name = "initial_profile_g#{}".format(g,'npy')
        np.save(file_name, psi_rd_ini)
    if (dat):
        file_name1 = "initial_profile_g#{}".format(g,'dat')
        np.savetxt(file_name1, psi_rd_ini)

    #plot data in Jupyter
    if PLOT:
        plt.plot(lr,lb, label='gravitational potential')
        plt.plot(lr,la, label='$\chi(r)$')
        plt.legend()
        plt.xlim(0.,(rge-1)*dr)

    print ('----Successfully initiated soliton profile----')

    return mass, beta


###########################simulation of soliton merger##############################

#----------function to check for soliton overlap (used for the case without self-interaction)-------------
def check_func(fix_one, solitons):
    for i in range(len(solitons)): #i: the total number of solitons
        m = max(fix_one[0], solitons[i][0]) #soliton data: [mass, position, velocity, phase, alpha]; fix_one is one fixed soliton;
        d_sol = 5.35854 / m #rc of a soliton with core mass m
        c_pos = np.array(fix_one[1])
        s_pos = np.array(solitons[i][1]) #the position of one soliton
        displacement = c_pos - s_pos
        distance = np.sqrt(displacement[0] ** 2 + displacement[1] ** 2 + displacement[2] ** 2) #the distance of two solitons
        if (distance < 2 * d_sol):
            return False
        else:
            return True
        #print("---- de Brogile length = %s ----"% (2 * np.pi / m/sum(abs(i) for i in solitons[i][2]))


def overlap_check(solitons): #which can be called in main code; 
    for k in range(len(solitons)):
        if (k != 0):
            if (not check_func(solitons[k], solitons[:k])): #suitable for multiply solitons
                print("----The initial soliton profile is too wide!----")
                print("----The first postions=%s----"%(k))
                return False
                break
            else:
                print("----The initial soliton profile is successfully set----")
                return True

#------creat grids in x and k spaces------------------
def grid_func(resol, length):
    #resol: the number of grids on each side in the real space
    #length: the normalized length of each side in the real space
    ncell = resol**3   # the number of cells     
    vcell = length**3 / ncell    # the normalized volume of each cell
    #-----------set up grids in the real space------------------------------------
    grid_vec = np.linspace( - length/2.0 + length/ float(2.0*resol), 
                       length/2.0 - length/float(2.0*resol), resol) # (-length/2, length/2) on each sides
    xarray, yarray, zarray = np.meshgrid(grid_vec, grid_vec, grid_vec, sparse=True, indexing ='ij') #(resol, 1,1) for x
    #dist = ne.evaluate("(x**2+y**2+z**2)**0.5")  # radial coordinates
    # adding arrays with different shapes via broadcasting


    #----------------set up grids in k space-----------------------------------
    kvec = 2 * np.pi * np.fft.fftfreq(resol, length / float(resol))       # dk = 2*pi/length
    kx, ky, kz = np.meshgrid(kvec, kvec, kvec, sparse=True, indexing='ij') # k vector; shape: (resol,1,1) for kx
    return xarray, yarray, zarray, kx, ky, kz

#--------function to put spherical soliton density profile into 3D box(Uses pre-computed array)----------------------
def ini_func(funct, xarray, yarray, zarray, position, alpha, f, dr):
    for index in np.ndindex(funct.shape): #interatively pass every gridpoints!
        # the distance of every gridpoint from the centre of the soliton
        distfromcentre = (
            (xarray[index[0], 0, 0] - position[0]) ** 2 +
            (yarray[0, index[1], 0] - position[1]) ** 2 +
            (zarray[0, 0, index[2]] - position[2]) ** 2 ) ** 0.5
        # Utilizing soliton profile array out to dimensionless radius 5.6
        # f is the pre-computated profile
        if (np.sqrt(alpha) * distfromcentre <= 6): #int: necessary for index
            funct[index] = alpha * f[int(np.sqrt(alpha) * (distfromcentre / dr + 1))]
        else:
            funct[index] = 0
    return funct

#ini_func_jit = nb.jit(ini_func) #speed up


#-------------create initial profiles----------------
def ini_merger(solitons, resol, xarray, yarray, zarray, psi, funct, f, dr, beta, t0):
    #if (not overlap_check(solitons)):
    #    print("----The initial soliton profile is too wide!----")
    #else:
    for s in solitons:
        phase = s[3]
        position = s[1]
        alpha = s[4]
        funct = ini_func(funct, xarray, yarray, zarray, position, alpha, f, dr)
        #Import velocity to solitons in a Galilean invariant way
        velx = s[2][0] #s[2]: velocity
        vely = s[2][1]
        velz = s[2][2] #t0: initial time
        funct = ne.evaluate("exp(1j*(alpha*beta*t0 + velx*xarray + vely*yarray + velz*zarray -0.5*(velx*velx+vely*vely+velz*velz)*t0 + phase))*funct")
        psi = ne.evaluate("psi + funct") #initial total wavefunction
    return psi

#---------------sponge boundary condition---------------------
def sponge_func(resol,length):
    func = np.zeros((resol, resol, resol), dtype="complex") #
    rn = length/2.0
    rp = 7/8 * rn
    rs = (rn + rp)/2.0
    delta = rn - rp
    v0 = 0.6
    xarray, yarray, zarray, kx, ky, kz = grid_func(resol, length)
    for index in np.ndindex(func.shape):
        dist = (xarray[index[0], 0, 0]** 2 + yarray[0, index[1], 0]** 2 + zarray[0, 0, index[2]]** 2 ) ** 0.5
        if dist >= rp:
            func[index] = -1j * v0 * (2 + np.tanh((dist-rs)/delta) - np.tanh(rs/delta) )
        else:
            func[index] = 0.0
    return func

#-------------------energy of system--------------------------------
def calculate_energies(resol, vell, g, egy_tot, mtot, egy_grav, 
    tot, egy_rest, psi, grav, ksq):
    egyarr = np.zeros((resol, resol, resol), dtype='float64')

    # Gravitational potential energy density of wdm
    egyarr = ne.evaluate('real(0.5*grav*real((abs(psi))**2))')
    egy_grav.append(vell * np.sum(egyarr))
    tot = vell * np.sum(egyarr)

    # other energy: kinematic + quantum + self-interaction
    funct = np.fft.fftn(psi)
    funct = ne.evaluate('-ksq*funct')
    funct = np.fft.ifftn(funct)
    egyarr = ne.evaluate('real(-0.5*conj(psi)*funct) + 0.5 * g * real(abs(psi))**4')
    egy_rest.append(vell * np.sum(egyarr))
    tot = tot + vell * np.sum(egyarr)

    egy_tot.append(tot) #total energy

    egyarr = ne.evaluate('real((abs(psi))**2)')
    mtot.append(vell * np.sum(egyarr))
    del egyarr, tot
    return egy_grav, egy_rest, egy_tot, mtot


def angular(psi, vcell,momenx,momeny,momenz, xarray, yarray, zarray, kx, ky, kz): #angular momentum

    xx = xarray[::,0,0]
    yy = yarray[0,::,0]
    zz = zarray[0,0,::]
    # nabla psi
    grad_psix, grad_psiy, grad_psiz = np.gradient(psi)
    # r times nabla psi
    crossx = np.einsum('k, ikl -> ikl', yy, grad_psiz) - np.einsum('l, ikl -> ikl', zz, grad_psiy)
    crossy = np.einsum('l, ikl -> ikl', zz, grad_psix) - np.einsum('i, ikl -> ikl', xx, grad_psiz)
    crossz = np.einsum('i, ikl -> ikl', xx, grad_psiy) - np.einsum('k, ikl -> ikl', yy, grad_psix)
    llx = ne.evaluate('real(-1j* conj(psi)*crossx)')
    lly = ne.evaluate('real(-1j* conj(psi)*crossy)')
    llz = ne.evaluate('real(-1j* conj(psi)*crossz)')

    momenx.append(vcell * np.sum(llx))
    momeny.append(vcell * np.sum(lly))
    momenz.append(vcell * np.sum(llz))
    del crossx, crossy, crossz, llx, lly, llz
    return momenx,momeny,momenz


#--------------------main program without BHs--------------------------------
def merger_sim(resol, length, Ntot, solitons, f, g, dr, beta, t0, psi, funct, ini_psi_plot, 
    ini_grav_plot, loop_initiate, actual_num_steps, save_num, sponge_bc, energy_check, angular_check):
    
    xarray, yarray, zarray, kx, ky, kz = grid_func(resol, length)
    ncell = resol**3   # the number of cells     
    vcell = length**3 / ncell    # the normalized volume of each cell
    ksq = ne.evaluate("kx**2+ky**2+kz**2") # magnitude of k**2; shape: (resol, resol, resol)
    psi = ini_merger(solitons, resol, xarray, yarray, zarray, psi, funct, f, dr, beta, t0)

    #av = np.sqrt(sum(np.real(np.abs(psi.reshape(ncell)))**2)*vcell) # normalization for wave function
    #psi = psi/av * np.sqrt(Ntot) # the true wave function in real space; (resol, resol, resol)

    #--------------check the output initial condition----------------------------
    amp = np.abs(psi).reshape(ncell)          #|psi|
    numdensity = amp**2                       #normalized number density of bosons
    avN = sum(amp**2)*vcell                   #each sample volume is ltot/ncell
    avn = sum(amp**2)/ncell                   #averaged number density
    jeanslength = 2*np.pi/((4*avn)**(1/4))    #jeanslength

    print("averaged number density (input) =", float(Ntot/length**3) )
    print("averaged number density (output) =",avn)
    print("total number of bosons (output) =",avN)
    print("jeanslength =",jeanslength)


    rhon = ne.evaluate("real(abs(psi)**2)")   # normalized number density; 'real' is necessary
    rhon_k = np.fft.fftn(rhon)    # Fourier transform of |psi|**2
    grav_k = np.divide(- rhon_k,  ksq)
    grav_k[0,0,0] = 0             # Avoid singularity of poisson equation at k = 0; 
    grav = np.real( np.fft.ifftn(grav_k) )   # initial self-gravitational potential


    if (ini_psi_plot):
        cmap_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, 256))
        cmap_colors[:, -1] = np.linspace(0, 1, 256)
        jet = mcolors.ListedColormap(cmap_colors)
        #-------------------------------2d-----------------------------------
        # sum along z slice
        data1 = np.abs(psi)**2
        datarho = data1.sum(2) #sum along z axis
        plt.imshow(datarho, cmap="jet",
           interpolation="nearest", origin="lower", norm=matplotlib.colors.LogNorm(1e-1, 1e+2),
          extent=[-length/2.0, length/2.0, -length/2.0, length/2.0])
        plt.colorbar(pad=0.1)
        plt.contour(datarho, levels=[1e+1,1e+2],
            extent=[- length/2.0, length/2.0, -length/2.0, length/2.0], 
            colors = "black", origin = 'lower', linewidths=0.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Initial number density $|\psi|^2$ for merger")
        plt.show()


    if (ini_grav_plot):
        #grav_amp = grav.reshape(ncell)
        grav_amp = grav.sum(2)
        #----------------------------------2d---------------------------------------
        # sum along z axis
        plt.imshow(grav_amp, cmap="jet", 
                   interpolation="nearest", origin="lower",
                   extent=[-length/2.0, length/2.0, -length/2.0, length/2.0])
        plt.colorbar(pad=0.1)
        plt.clim(-2,10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Initial Self-Gravitational Potential $\Phi$ for merger")
        plt.show()


    #calculate the intial energy and angular momentum
        
    egy_tot= []
    mtot = []
    egy_grav = []
    tot = []
    egy_rest = []
    egy_grav, egy_rest, egy_tot, mtot \
        = calculate_energies(resol, vcell, g, egy_tot, mtot, egy_grav, 
                tot, egy_rest, psi, grav, ksq)

    print("egy_grav_ini = ", egy_grav)
    print("egy_rest_ini = ", egy_rest)
    print("egy_tot_ini = ", egy_tot)
    print("mtot_ini = ", mtot)

    # calculate the angular momentum
    #if angular_check:
    momenx = []
    momeny = []
    momenz = []
    momenx, momeny,momenz \
        = angular(psi, vcell,momenx,momeny,momenz, xarray, yarray, zarray, kx, ky, kz)

    print("momenx_ini = ", momenx)
    print("momeny_ini = ", momeny)
    print("momenz_ini = ", momenz)


    # loop start: pesudo-spectral method
    if (loop_initiate):
        #---------------Time step--------------------
        h = (length/resol)**2/6 # satisfies CFL condition
        print("the renormalized time step =",h)

        #-------------------Loop-----------------------------
        for ix in range(actual_num_steps): 
            psi = ne.evaluate("exp(-1j * 0.5 * h * g * real(abs(psi))**2 ) * psi")
            psi = ne.evaluate("exp(-1j * 0.5 * h * (grav + sponge_bc) ) * psi")
            funct = np.fft.fftn(psi)
            funct = ne.evaluate("exp(-1j * 0.5 * h * ksq) * funct")
            psi = np.fft.ifftn(funct)
            rho = ne.evaluate("real(abs(psi)**2)")
            rho_k = np.fft.fftn(rho)                  
            grav_k = np.divide(-rho_k,  ksq)
            grav_k[0, 0, 0] = 0
            grav = np.real( np.fft.ifftn(grav_k) )
            psi = ne.evaluate("exp(-1j * 0.5 * h * g * real(abs(psi))**2) * psi")
            psi = ne.evaluate("exp(-1j * 0.5 * h * (grav + sponge_bc) ) * psi")

            if energy_check:
                egy_grav, egy_rest, egy_tot, mtot \
                =calculate_energies(resol, vcell, g, egy_tot, mtot, egy_grav, 
                    tot, egy_rest, psi, grav, ksq)

            if angular_check:
                momenx, momeny,momenz \
                = angular(psi, vcell,momenx,momeny,momenz, xarray, yarray, zarray, kx, ky, kz)
                        
            # create a unique file path
            if ix%save_num==0:
                file_name = "psi_#{0}".format(ix,'npy')
                np.save(os.path.join('merger_data', file_name), psi)

        if energy_check:
            np.save(os.path.join('mergerENERGY_data', "egy_grav.npy"), np.array(egy_grav))
            np.save(os.path.join('mergerENERGY_data', "egy_rest.npy"), np.array(egy_rest))
            np.save(os.path.join('mergerENERGY_data', "egy_tot.npy"), np.array(egy_tot))
            np.save(os.path.join('mergerENERGY_data', "mtot.npy"), np.array(mtot))

        if angular_check:
            np.save(os.path.join('mergerENERGY_data', "momx.npy"), np.array(momenx))
            np.save(os.path.join('mergerENERGY_data', "momy.npy"), np.array(momeny))
            np.save(os.path.join('mergerENERGY_data', "momz.npy"), np.array(momenz))



#----------------embed test masses into 3D grid (abandoned in new version)------------
def convert_bh(r1, r2, resol, length, xarray, yarray, zarray,factor):
    for index in np.ndindex( (resol, resol, resol) ):
        dist1 = (
            (xarray[index[0], 0, 0] - r1[0]) ** 2 +
            (yarray[0, index[1], 0] - r1[1]) ** 2 +
            (zarray[0, 0, index[2]] - r1[2]) ** 2 ) ** 0.5
        dist2 = (
            (xarray[index[0], 0, 0] - r2[0]) ** 2 +
            (yarray[0, index[1], 0] - r2[1]) ** 2 +
            (zarray[0, 0, index[2]] - r2[2]) ** 2 ) ** 0.5
        if dist1 <= length/resol*factor:
            print("BH1: ", (index[0], index[1], index[2]) )
        elif dist2 <= length/resol*factor:
            print("BH2: ", (index[0], index[1], index[2]) )


#---------------interploration function-------------------------
def interp_func(data, length, x, y, z):
    """
    Interpolate a 3D array of data at the points (x, y, z) using linear interpolation.
    
    Args:
        data (ndarray): The 3D array of data to interpolate.
        x (ndarray): The x-coordinates of the points to interpolate at.
        y (ndarray): The y-coordinates of the points to interpolate at.
        z (ndarray): The z-coordinates of the points to interpolate at.
        
    Returns:
        The interpolated values of the data at the points (x, y, z).
    """
    # get the shape of the data array
    nx, ny, nz = data.shape
    
    # create arrays of x, y, and z coordinates
    xxx = np.linspace(-length/2, length/2, nx)
    yyy = np.linspace(-length/2, length/2, ny)
    zzz = np.linspace(-length/2, length/2, nz)
    
    # create the interpolation function
    interfunc = RegularGridInterpolator((xxx, yyy, zzz), data, method="linear")
    
    # evaluate the interpolation function at the specified points
    vals = interfunc(np.column_stack((x, y, z)))[0]
    
    return vals


#--------------------BH dynamics-------------------------
def eom_bh(t, y, coeffi, grad, length): #Newton's gravity
    [r1x, r1y, r1z, v1x, v1y, v1z, r2x, r2y, r2z, v2x, v2y, v2z] = y
    [bh_m1, bh_m2] = coeffi
    r21 = np.array([r2x-r1x, r2y-r1y, r2z-r1z])
    if np.linalg.norm(r21) < 1e-8:
        print("-----Two BHs collide!-----") #avoid the overlap
        sys.exit()
    else:
        fx1 = interp_func(grad[0], length, r1x, r1y, r1z)
        fy1 = interp_func(grad[1], length, r1x, r1y, r1z)
        fz1 = interp_func(grad[2], length, r1x, r1y, r1z)
        fx2 = interp_func(grad[0], length, r2x, r2y, r2z)
        fy2 = interp_func(grad[1], length, r2x, r2y, r2z)
        fz2 = interp_func(grad[2], length, r2x, r2y, r2z)
        return np.array([
            v1x, v1y, v1z,
            bh_m2 / np.linalg.norm(r21)**3/4/np.pi * r21[0] - fx1,
            bh_m2 / np.linalg.norm(r21)**3/4/np.pi * r21[1] - fy1,
            bh_m2 / np.linalg.norm(r21)**3/4/np.pi * r21[2] - fz1,
            v2x, v2y, v2z,
            - bh_m1 / np.linalg.norm(r21)**3/4/np.pi * r21[0] - fx2,
            - bh_m1 / np.linalg.norm(r21)**3/4/np.pi * r21[1] - fy2,
            - bh_m1 / np.linalg.norm(r21)**3/4/np.pi * r21[2] - fz2
            ])

#--------------------main program with BHs--------------------------------
def merger_sim_bh(coeffi, y_val, resol, length, Ntot, solitons, f, dr, beta, t0, psi, funct, ini_psi_plot, 
    ini_grav_plot, loop_initiate, actual_num_steps, save_num, sponge_bc, energy_check, angular_check):
    
    xarray, yarray, zarray, kx, ky, kz = grid_func(resol, length)
    ncell = resol**3   # the number of cells     
    vcell = length**3 / ncell    # the normalized volume of each cell
    ksq = ne.evaluate("kx**2+ky**2+kz**2") # magnitude of k**2; shape: (resol, resol, resol)
    psi = ini_merger(solitons, resol, xarray, yarray, zarray, psi, funct, f, dr, beta, t0)

    #av = np.sqrt(sum(np.real(np.abs(psi.reshape(ncell)))**2)*vcell) # normalization for wave function
    #psi = psi/av * np.sqrt(Ntot) # the true wave function in real space; (resol, resol, resol)

    #--------------check the output initial condition----------------------------
    amp = np.abs(psi).reshape(ncell)          #|psi|
    numdensity = amp**2                       #normalized number density of bosons
    avN = sum(amp**2)*vcell                   #each sample volume is ltot/ncell
    avn = sum(amp**2)/ncell                   #averaged number density
    jeanslength = 2*np.pi/((4*avn)**(1/4))    #jeanslength

    print("averaged number density (input) =", float(Ntot/length**3) )
    print("averaged number density (output) =",avn)
    print("total number of bosons (output) =",avN)
    print("jeanslength =",jeanslength)


    rhon = ne.evaluate("real(abs(psi)**2)")   # normalized number density; 'real' is necessary
    rhon_k = np.fft.fftn(rhon)    # Fourier transform of |psi|**2
    grav_k = np.divide(- rhon_k,  ksq)
    grav_k[0,0,0] = 0             # Avoid singularity of poisson equation at k = 0; 
    grav = np.real( np.fft.ifftn(grav_k) )   # initial self-gravitational potential


    if (ini_psi_plot):
        cmap_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, 256))
        cmap_colors[:, -1] = np.linspace(0, 1, 256)
        jet = mcolors.ListedColormap(cmap_colors)
        #-------------------------------2d-----------------------------------
        # sum along z slice
        data1 = np.abs(psi)**2
        datarho = data1.sum(2) #sum along z axis
        plt.imshow(datarho, cmap="jet",
           interpolation="nearest", origin="lower", norm=matplotlib.colors.LogNorm(1e-1, 1e+2),
          extent=[-length/2.0, length/2.0, -length/2.0, length/2.0])
        plt.colorbar(pad=0.1)
        plt.contour(datarho, levels=[1e-1, 1e+0],
            extent=[- length/2.0, length/2.0, -length/2.0, length/2.0], 
            colors = "black", origin = 'lower', linewidths=0.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Initial number density $|\psi|^2$ for merger")
        plt.plot(y_val[1], y_val[0], 'bo', markersize=3) #swap x and y
        plt.plot(y_val[7], y_val[6], 'go', markersize=3)
        plt.show()


    if (ini_grav_plot):
        #grav_amp = grav.reshape(ncell)
        grav_amp = grav.sum(2)
        #----------------------------------------2d-----------------------------------------
        # sum along z axis
        plt.imshow(grav_amp, cmap="jet", 
                   interpolation="nearest", origin="lower",
                   extent=[-length/2.0, length/2.0, -length/2.0, length/2.0])
        plt.colorbar(pad=0.1)
        plt.clim(-2,10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Initial Self-Gravitational Potential $\Phi$ for merger")
        plt.show()

    #calculate the intial energy and angular momentum
        
    egy_tot= []
    mtot = []
    egy_grav = []
    tot = []
    egy_rest = []
    egy_grav, egy_rest, egy_tot, mtot \
        = calculate_energies(resol, vcell, g, egy_tot, mtot, egy_grav, 
                tot, egy_rest, psi, grav, ksq)

    print("egy_grav_ini = ", egy_grav)
    print("egy_rest_ini = ", egy_rest)
    print("egy_tot_ini = ", egy_tot)
    print("mtot_ini = ", mtot)

    # calculate the angular momentum
    #if angular_check:
    momenx = []
    momeny = []
    momenz = []
    momenx, momeny,momenz \
        = angular(psi, vcell,momenx,momeny,momenz, xarray, yarray, zarray, kx, ky, kz)

    print("momenx_ini = ", momenx)
    print("momeny_ini = ", momeny)
    print("momenz_ini = ", momenz)


    # loop start: pesudo-spectral method
    if (loop_initiate):
        #---------------Time step--------------------
        h = (length/resol)**2/6 # satisfies CFL condition
        print("the renormalized time step =",h)
        
        t = 0.0
        #-------------------Loop-----------------------------
        for ix in range(actual_num_steps):  
            psi = ne.evaluate("exp(-1j * 0.5 * h * (grav + sponge_bc) ) * psi")
            funct = np.fft.fftn(psi)
            funct = ne.evaluate("exp(-1j * 0.5 * h * ksq) * funct")
            psi = np.fft.ifftn(funct)
            rho = ne.evaluate("real(abs(psi)**2)")
            rho_k = np.fft.fftn(rho)                  
            grav_k = np.divide(-rho_k,  ksq)
            grav_k[0, 0, 0] = 0
            grav = np.real( np.fft.ifftn(grav_k) )
            psi = ne.evaluate("exp(-1j * 0.5 * h * (grav + sponge_bc) ) * psi")
            # gravitational potential
            grad_grav = np.gradient(grav) # list: gradient of field

            if energy_check:
                egy_grav, egy_rest, egy_tot, mtot \
                =calculate_energies(resol, vcell, g, egy_tot, mtot, egy_grav, 
                    tot, egy_rest, psi, grav, ksq)

            if angular_check:
                momenx, momeny,momenz \
                = angular(psi, vcell,momenx,momeny,momenz, xarray, yarray, zarray, kx, ky, kz)
        
            #rk4 for BH dynamics
            k1 = h * eom_bh( t,             y_val, coeffi, grad_grav, length)
            k2 = h * eom_bh( t + 1/4 * h,   y_val + 1/4 * k1, coeffi, grad_grav, length)
            k3 = h * eom_bh( t + 3/8 * h,   y_val + 3/32 * k1 + 9/32 * k2, coeffi, grad_grav, length)
            k4 = h * eom_bh( t + 12/13 * h, y_val + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3, coeffi, grad_grav, length)
            k5 = h * eom_bh( t + h,         y_val + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4, coeffi, grad_grav, length)
            #k6 = h * eom_bh( t + 1/2 * h,   y - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5, coeffi, grad_grav)

            rk4 = y_val + 25/216 * k1 + 1408/2565 * k3 + 2197/4101 * k4 - 1/5 * k5
            #rk5 = y + 16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6
            
            y_val = np.array([rk4[0], rk4[1], rk4[2],
                rk4[3], rk4[4], rk4[5], 
                rk4[6], rk4[7], rk4[8],
                rk4[9], rk4[10], rk4[11]])
            t += h

            # create a unique file path
            if ix%save_num==0:
                file_name = "psi_#{}".format(ix,'npy')
                np.save(os.path.join('merger_data', file_name), psi)

                r1 = np.array([rk4[0], rk4[1], rk4[2]]) # store data
                v1 = np.array([rk4[3], rk4[4], rk4[5]])
                r2 = np.array([rk4[6], rk4[7], rk4[8]])
                v2 = np.array([rk4[9], rk4[10], rk4[11]])
                bh_data = np.stack([r1, v1, r2, v2], axis=0)
                file_name1 = "bh_#{}".format(ix,'npy')
                np.save(os.path.join('mergerBH_data', file_name1), bh_data)
                del r1, v1, r2, v2, bh_data # release memory

        if energy_check:
            np.save(os.path.join('mergerENERGY_data', "egy_grav.npy"), np.array(egy_grav))
            np.save(os.path.join('mergerENERGY_data', "egy_rest.npy"), np.array(egy_rest))
            np.save(os.path.join('mergerENERGY_data', "egy_tot.npy"), np.array(egy_tot))
            np.save(os.path.join('mergerENERGY_data', "mtot.npy"), np.array(mtot))

        if angular_check:
            np.save(os.path.join('mergerENERGY_data', "momx.npy"), np.array(momenx))
            np.save(os.path.join('mergerENERGY_data', "momy.npy"), np.array(momeny))
            np.save(os.path.join('mergerENERGY_data', "momz.npy"), np.array(momenz))
        



#####################soliton formation simulation########################

def soliton_formaiton(resol, length, Ntot,ini_psi_plot, 
    ini_grav_plot, loop_initiate, actual_num_steps, save_num,sponge_bc):
    #resol: the number of grids on each side in the real space
    #length: the normalized length of each side in the real space
    #Ntot: the normalized total number of bosons: N = number density * length**3
    
    xarray, yarray, zarray, kx, ky, kz = grid_func(resol, length)
    ksq = ne.evaluate("kx**2+ky**2+kz**2") # magnitude of k**2; shape: (resol, resol, resol)
    ncell = resol**3   # the number of cells     
    vcell = length**3 / ncell    # the normalized volume of each cell

    #-------------------initial Gaussian distribution--------------------------------
    phase = np.random.uniform(0., 2.*np.pi, ksq.shape) # random phase; (resol, resol, resol)
    psi_k = np.sqrt(np.exp(-ksq))*np.exp(1j*phase)     # initial Gaussian wave
    psi = np.fft.ifftn(psi_k)                          # inverse Fourier transform
    av = np.sqrt(sum(np.real(np.abs(psi.reshape(ncell)))**2)*vcell) # normalization for wave function
    psi = psi/av * np.sqrt(Ntot)                       # the true wave function in real space; (resol, resol, resol)

    #--------------check the output initial condition----------------------------
    amp = np.abs(psi).reshape(ncell) # |psi|
    numdensity = amp**2                       # normalized number density of bosons
    avN = sum(amp**2)*vcell                   # each sample volume is ltot/ncell
    avn = sum(amp**2)/ncell                   # averaged number density
    jeanslength = 2*np.pi/((4*avn)**(1/4))    # jeanslength

    print("averaged number density (input) =", float(Ntot/length**3) )
    print("averaged number density (output) =",avn)
    print("total number of bosons (output) =",avN)
    print("jeanslength =",jeanslength)

    if (ini_psi_plot):
        cmap_colors = plt.cm.get_cmap('jet')(np.linspace(0, 1, 256))
        cmap_colors[:, -1] = np.linspace(0, 1, 256)
        jet = mcolors.ListedColormap(cmap_colors)
        #-------------------------------2d-----------------------------------
        # z=0 slice;
        plt.imshow((numdensity[:resol**2]).reshape((resol,resol)), cmap="jet",
           interpolation="nearest", origin="lower", norm=matplotlib.colors.LogNorm(1e-1, 1e+0),
          extent=[-length/2.0, length/2.0, -length/2.0, length/2.0])
        plt.colorbar(pad=0.1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Initial number density $|\psi|^2$")
        plt.show()

        #-------------------3d-------------------TURN OFF the sparse!
        #fig = plt.figure(dpi=200)
        #ax = fig.add_subplot(111, projection="3d")
        #num3d = numdensity.reshape((resol,resol,resol))
        #num3dimg = ax.scatter(x, y, z, c=num3d, s=0.01, alpha=0.01, marker='o',norm=matplotlib.colors.LogNorm(1e+0, 1e+3),cmap="jet")
        #plt.colorbar(num3dimg,shrink=0.5,cmap="jet",alpha=1.0)
        #plt.show()

        # why plot this?
        #psitest = np.abs(np.fft.fftshift(np.fft.fftn(psi)))
        #lt.imshow(psitest[3], cmap="jet", interpolation="nearest", origin="lower",
        #         extent=[-length/2.0, length/2.0, -length/2.0, length/2.0])

    if (ini_grav_plot):
        rhon = ne.evaluate("real(abs(psi)**2)")   # normalized number density; 'real' is necessary
        rhon_k = np.fft.fftn(rhon)    # Fourier transform of |psi|**2
        grav_k = np.divide(- rhon_k,  ksq)
        grav_k[0,0,0] = 0             # Avoid singularity of poisson equation at k = 0; 
        grav = np.real( np.fft.ifftn(grav_k) )   # initial self-gravitational potential
        grav_amp = grav.reshape(ncell)

        #----------------------------------------2d-----------------------------------------
        # z=0 slice
        plt.imshow((grav_amp[:resol**2]).reshape((resol,resol)), cmap="jet", 
                   interpolation="nearest", origin="lower",
                   extent=[-length/2.0, length/2.0, -length/2.0, length/2.0])
        plt.colorbar(pad=0.1)
        plt.clim(-1,10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Initial Self-Gravitational Potential $\Phi$")
        plt.show()

        #---------------------3d------------------TURN OFF the sparse!
        #fig = plt.figure(dpi=200)
        #ax = plt.axes(projection="3d")
        #grav3d = grav_amp.reshape((resol,resol,resol))
        #grav3dimg = ax.scatter(x, y, z, c=grav3d, s=0.01, alpha=0.01, marker='o',cmap="jet")
        #plt.colorbar(grav3dimg,shrink=0.5,cmap="jet")
        #plt.show()
        #norm=matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.01,vmin=-1e-3, vmax=2e-2, base=10)

    # loop start: pesudo-spectral method
    if (loop_initiate):
        #---------------Time step--------------------
        h = (length/resol)**2/6 # satisfies CFL condition
        print("the renormalized time step =",h)

        #-------------------Loop-----------------------------
        for ix in range(actual_num_steps):  
            #psi = ne.evaluate("exp(-1j * 0.5 * h * grav) * psi")
            psi = np.exp(-1j * 0.5 * h * (grav + sponge_bc) ) * psi
            funct = np.fft.fftn(psi)
            funct = ne.evaluate("exp(-1j * 0.5 * h * ksq) * funct")
            psi = np.fft.ifftn(funct)
            rho = ne.evaluate("real(abs(psi)**2)")
            rho_k = np.fft.fftn(rho)                  
            grav_k = np.divide(-rho_k,  ksq)
            grav_k[0, 0, 0] = 0
            grav = np.real( np.fft.ifftn(grav_k) )
            #psi = ne.evaluate("exp(-1j * 0.5 * h * grav) * psi")
            psi = np.exp(-1j * 0.5 * h * (grav + sponge_bc) ) * psi
            
            # create a unique file path
            if ix%save_num==0:
                file_name = "psi_#{0}".format(ix,'npy')
                np.save(os.path.join('soliton_data', file_name), psi)