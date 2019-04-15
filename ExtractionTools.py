# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:13:26 2018

@author: Isidoro
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

from scipy.stats import chi2
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

# == Data Extraction ==========================================================
def extract_RadonData(filePath=None):
    """
    Esta funcion extrae la informacion sobre un experimento de exhalacion,
    devolviendo un dataframe con los datos obtenidos.
    
    La ruta dada por "filePath" debe ser la del archivo correspondiente, NO la
    de la carpeta.
    """
    ## File handling 
    filePath = open_FileOrDir(filePath,kind='file')
        
    ## Experiment info handling ###############################################
    # Obtain specific information about the experiment
    experiment_info = os.path.basename(filePath).split('.')[0].split('_')
    if len(experiment_info) == 5:
        experiment_info.append('None')
    # R7 and R7Sniff are treated the same
    if experiment_info[1] == 'R7Sniff':
        experiment_info[1] = 'R7'
    ###########################################################################
    
    ## Assertions #############################################################
    # Define valid devices and chambers
    valid_devices = ['aG','aC','aGv2','aCv2','R7','R7v2','R7DRY','RS']
    valid_chambers = ['Cnt','Cn2','V10','V15','V35',
                      'C0','C3','C6','C9','C12','C14']    
    # Make sure the device of the experiment can be handled
    assert experiment_info[1] in valid_devices   
    ###########################################################################
    
    ## Definition of constants ################################################
    # Define a name list for the columns
    names = ['Date','Radon','Radon Error']
    ## Define the relevant columns for each possible device
    # Alphaguard Huelva
    cols_aG = [0,1,2]
    # Alphaguard Barcelona
    cols_aC = [0,1,2,3]
    # Rad 7
    cols_R7 = [0,3,4]
    # Radon Scout
    cols_RS = [0,1,2]
    ###########################################################################
    
    ## Extraction #############################################################
    # Check which device is and extract its data
    try:
        if experiment_info[1] in ['aG','aGv2']:
            # Extract data
            df = pd.read_csv(filePath,
                             sep='\t', skiprows=2, decimal=',',
                             header=None, usecols=cols_aG,index_col=0,
                             names=names,parse_dates=True, dayfirst=True)
            # Extract header
            header = pd.read_csv(filePath,
                                 sep='\t', skiprows=0, nrows=1,
                                 header=None, usecols=[1])
            # Extract units
            units = str(header.values[0][0]).split(' ')[1]
            # Fix order of magnitude
            if units == '(KBq/m3)':
                df['Radon'] *= 1000
            elif units == '(MBq/m3)':
                df['Radon'] *= 1000000
                
        if experiment_info[1] in ['R7','R7v2','R7DRY']:
            df = pd.read_csv(filePath,
                             sep='\t', skiprows=3, decimal=',',
                             header=None, usecols=cols_R7,index_col=0,
                             names=names,parse_dates=True, dayfirst=True)
        if experiment_info[1] == 'RS':
            df = pd.read_csv(filePath,
                             sep='\t', skiprows=19, decimal=',',
                             header=None, usecols=cols_RS,index_col=0,
                             names=names,parse_dates=True, dayfirst=True)
            # El error del RS viene en porcentaje, lo pasamos a absoluto
            df['Radon Error'] = df['Radon']*df['Radon Error']/100
        if experiment_info[1] in ['aC','aCv2']:               
            df= pd.read_csv(filePath,
                            sep='\s+', skiprows=4, decimal=',',
                            header=None, usecols=cols_aC)
            # Generamos los indices a partir de la fecha y la hora
            idx = pd.to_datetime(df[0]+' '+ df[1],dayfirst=True)
            # Reasignamos el indice
            df = df.set_index(idx)
            # Eliminamos las columnas sobrantes
            df = df.iloc[:,2:]
            # Reasignamos los nombres
            df.columns = names[1:]
    except:
        print('# ======================================== #')
        print(filePath)
        print('# ======================================== #')
        raise
    ###########################################################################
    
    ## Return results #########################################################
    return (df,experiment_info)

def extract_AmbData(filePath=None):
    """
    Esta funcion extrae la informacion de temperatura y humedad de un archivo
    de EL-USB.
    
    La ruta dada por "filePath" debe ser la del archivo correspondiente, NO la
    de la carpeta.
    """
    ## File handling 
    filePath = open_FileOrDir(filePath,kind='file')
        
    ## Experiment info handling ###############################################
    # Obtain specific information about the experiment
    experiment_info = os.path.basename(filePath).split('.')[0].split('_')
    if len(experiment_info) == 5:
        experiment_info.append('None')
    # R7 and R7Sniff are treated the same
    if experiment_info[1] == 'R7Sniff':
        experiment_info[1] = 'R7'
    elif experiment_info[1] in ['EL1','EL2','EL3']:
        experiment_info[1] = 'EL'
    ###########################################################################
    
    ## Assertions #############################################################
    # Define valid devices and chambers
    valid_devices = ['aG','aC','R7','RS','EL']
#    valid_chambers = ['Cnt','Cn2','V10','V15','V35',
#                      'C0','C3','C6','C9','C12','C14']    
    # Make sure the device of the experiment can be handled
    assert experiment_info[1] in valid_devices   
    ###########################################################################
    
    ## Definition of constants ################################################
    ## Define the relevant columns for each possible device and its names
    # Alphaguard Huelva
    cols_aG = [0,5,6,7]
    names_aG = ['Date','Temperature','Pressure','Relative Humidity']
    # Alphaguard Barcelona
    cols_aC = [0,1,8,9,10]
    names_aC = ['Date','Temperature','Pressure','Relative Humidity']
    # Rad 7
    cols_R7 = [0,1,2]
    names_R7 = ['Date','Temperature','Relative Humidity']
    # Radon Scout
    cols_RS = [0,3,4,5]
    names_RS = ['Date','Temperature','Relative Humidity','Pressure']
    # Radon Scout
    cols_EL = [1,2,3]
    names_EL = ['Date','Temperature','Relative Humidity']
    ###########################################################################
    
    ## Extraction #############################################################
    # Check which device is and extract its data
    try:
        if experiment_info[1] == 'aG':
            # Extract data
            df = pd.read_csv(filePath,
                             sep='\t', skiprows=2, decimal=',',
                             header=None, usecols=cols_aG,index_col=0,
                             names=names_aG,parse_dates=True, dayfirst=True)                
        elif experiment_info[1] == 'R7':
            df = pd.read_csv(filePath,
                             sep='\t', skiprows=3, decimal='.',
                             header=None, usecols=cols_R7,index_col=0,
                             names=names_R7,parse_dates=True, dayfirst=True)
        elif experiment_info[1] == 'RS':
            df = pd.read_csv(filePath,
                             sep='\t', skiprows=19, decimal=',',
                             header=None, usecols=cols_RS,index_col=0,
                             names=names_RS,parse_dates=True, dayfirst=True)
        elif experiment_info[1] == 'aC':               
            df= pd.read_csv(filePath,
                            sep='\s+', skiprows=4, decimal=',',
                            header=None, usecols=cols_aC)
            # Generamos los indices a partir de la fecha y la hora
            idx = pd.to_datetime(df[0]+' '+ df[1],dayfirst=True)
            # Reasignamos el indice
            df = df.set_index(idx)
            # Eliminamos las columnas sobrantes
            df = df.iloc[:,2:]
            # Reasignamos los nombres
            df.columns = names_aC[1:]
            df.index.name = 'Date'
        elif experiment_info[1] == 'EL':               
            df= pd.read_csv(filePath,
                            sep=',', skiprows=1, decimal='.',
                            header=None,usecols=cols_EL,names=names_EL,
                            index_col=0,parse_dates=True)
#            # Reasignamos el indice
#            df = df.set_index(df['Date']).drop('Date',axis=1)
    except:
        print('# ======================================== #')
        print(filePath)
        print('# ======================================== #')
        raise
    ###########################################################################
    
    ## Return results #########################################################
    return df
# =============================================================================

# == Compute the fit ==========================================================
def compute_fit(df,fitType='expGen',t_initial=10,t_duration=None,
                initial_parameters=None):
    """
    This function takes in a DataFrame with radon data and its error and 
    computes a fit.
    
    Type of fit, duration and initial parameters can be configured using the
    function parameters.
    """
    ## Definition of functions ################################################
    ## linear fit
    def f_lin(t,a,b):
        return a*t+b
    # Define initial parameters and boundaries
    p_lin = [1e0,1e1]
    n_lin = 2
    bounds_lin = ([0,-1e4],[1e6,1e6])
    ## Parabolic
    def f_par(t,a,b,c):
        return a*t+b+c*t**2
    # Define initial parameters and boundaries
    p_par = [1e0,1e1,-1e-5]
    n_par = 3
    bounds_par = ([0,-1e4],[1e6,1e6],[0,-1e10])
    ## Define exponential fit with zero initial concentration
    def f_exp(t,a,b):
        return a*(1-np.exp(-b*t))
    # Define initial parameters and boundaries
    p_exp = [1e3,1e-6]
    n_exp = 2
    bounds_exp = ([0,0],[1e8,1])
    ## Define general exponential fit 
    def f_expGen(t,a,b,c):
        return a+b*np.exp(-c*t)
    # Define initial parameters and boundaries
    p_expGen = [1e3,1e3,1e-4]
    n_expGen = 3
    bounds_expGen = (([0,-1e8 ,0],[1e8,1e8,1])) 
    # Definimos un diccionario para acceder a las funciones y sus parametros 
    fitParams = {'lin':(f_lin,p_lin,n_lin,bounds_lin),
                 'par':(f_par,p_par,n_par,bounds_par),
                 'exp':(f_exp,p_exp,n_exp,bounds_exp),
                 'expGen':(f_expGen,p_expGen,n_expGen,bounds_expGen)}
    ###########################################################################

    ## Define variables to fit ################################################
    # Calculamos el espaciado temporal entre valores 
    t_step = (df.index.values[1] - df.index.values[0])
    # Calculamos los segundos que ha durado el experimento
    t_Raw = df.index.values - df.index.values[0] + t_step
    # Lo pasamos a segundos
    t_Raw = t_Raw/np.timedelta64(1,'s')
    # Cogemos los datos de radon y su error
    Rn_Raw = np.array(df['Radon'])
    RnE_Raw = np.array(df['Radon Error'])
    # Buscamos los indices de t que superen a t_init (*60: min -> s)
    i = np.where(t_Raw>t_initial*60)
    # Generamos los valores que se usaran en el ajuste
    t = t_Raw[i]
    Rn = Rn_Raw[i]
    RnE = RnE_Raw[i]
    # Si tenemos un valor para la duracion del experimento
    if t_duration:
        # Sumamos la duracion al inicio (*60: min -> s)
        t_fit = (t_initial + t_duration)*60
        # Buscamos los valores de t menores que t_fit
        i = np.where(t<t_fit)
        # Generamos los valores que se usaran en el ajuste
        t = t[i]
        Rn = Rn[i]
        RnE = RnE[i]  
    ###########################################################################

    ## Apply the fit ##########################################################
    # Ajuste
    try:
        # Check if we have initial parameters
        if initial_parameters:
            assert len(initial_parameters) == len(fitParams[fitType][1])
            p0 = initial_parameters
            popt, pcov = curve_fit(fitParams[fitType][0],t,Rn,
                       sigma=RnE,absolute_sigma=True,
                       p0=p0)
        else:
            p0 = fitParams[fitType][1]
            popt, pcov = curve_fit(fitParams[fitType][0],t,Rn,
                                   sigma=RnE,absolute_sigma=True,
                                   p0=p0)
#    except (RuntimeError,ValueError,TypeError,error) as inst:
    except:
        print(df.index.values[0])
        # Llenamos los parametros del ajuste con nans
        popt = np.array([np.nan]*fitParams[fitType][2])
        # Creamos una matriz de covarianza vacia
        pcov = np.empty((fitParams[fitType][2],fitParams[fitType][2]))
        # La llenamos con nan
        pcov.fill(np.nan)
        # Calculamos el resto de parametros
        perr = np.sqrt(np.diag(pcov))
        R2adj = np.nan
        chi = np.nan
        pValue = np.nan
        # Guaradmos el error
#        fit_error = inst
#        print(inst)
    else:
        # No error has occured
#        fit_error = None
        # Compute uncertainties 
        perr = np.sqrt(np.diag(pcov))        
        # Compute R2 and R2adj
        R2 = 1 - (np.sum((Rn-fitParams[fitType][0](t,*popt))**2)/
                  np.sum((Rn-np.mean(Rn))**2))
        R2adj = (1 - ((1-R2)*(len(t)-1)/(len(t)-1-1)))
        
        # Compute chi-squared
        chi = np.sum((Rn-fitParams[fitType][0](t,*popt))**2/RnE**2)
        # freedom degree
        n = len(t)-fitParams[fitType][2]-1
        # -value
        pValue = chi2.sf(chi,n)
        #######################################################################
    
    ## Return fit results #####################################################
#    return [fitType,popt,perr,R2adj,chi,pValue,fit_error]
    return [fitType,popt,perr,R2adj,chi,pValue]
    ###########################################################################
# =============================================================================
    
# == Compute exhalation =======================================================
def compute_exhalation(fit_results,experiment_info):
    """
    This function takes in the results of curve fitting done by "compute_fit()"
    and calculates exhalation.
    """
    ## Input procesation ######################################################
    # fit_results
    fitType = fit_results[0]
    popt = fit_results[1]
    perr = fit_results[2]
    # Make sure R7Sniff if converted to R7
    if experiment_info[1] == 'R7Sniff':
        experiment_info[1] = 'R7'
    ###########################################################################
  
    ## Definicion de Areas y Volumenes ########################################
    # Volumen aparatos (R7 supone que la drierita no ocupa volumen) (m3)
    V_ap = {'aG':6E-4+(1.11+0.156+0.682+0.44)*np.pi*0.003**2+0.13*np.pi*0.002**2,
            'aC':6E-4+(0.87+0.18+0.70+0.38)*np.pi*0.003**2+0.21*np.pi*0.002**2,
            'aGv2':(6E-4 #Alphaguard
                    +(0.0667+0.0067)*np.pi*0.0015**2+(0.0145)*np.pi*0.0025**2 #Chamber->bomba
                    +(0.0373)*np.pi*0.0015**2 #bomba->aG
                    +(0.001)*np.pi*0.0015**2+(0.0767)*np.pi*0.0025**2), #aG->chamber
            'aCv2':(6E-4 #Alphaguard
                    +(0.0667+0.0067)*np.pi*0.0015**2+(0.0145)*np.pi*0.0025**2 #Chamber->bomba
                    +(0.0373)*np.pi*0.0015**2 #bomba->aG
                    +(0.001)*np.pi*0.0015**2+(0.0767)*np.pi*0.0025**2), #aG->chamber
            'R7':(1E-3+(0.666+0.764+0.04)*np.pi*0.003**2
                      +(0.095+0.05)*np.pi*0.002**2+(0.135+0.036)*np.pi*0.004**2
                      +0.28*np.pi*0.0295**2),
            'R7v2':(1E-3# R7
                    +np.pi*(0.9*0.002**2+0.07*0.0015**2+0.06*0.0035**2) # Chamber->Drierita
                    +np.pi*(0.28*0.0295**2) # Drierita
                    +np.pi*(0.89*0.002**2+(0.06+0.07+0.08)*0.0015**2) # Drierita->R7
                    +np.pi*(0.9*0.0015**2)), # R7->Chamber
            'R7DRY':(1E-3 #R7
                     +(34+47+52)*1E-6 #DRYSTIK (Pump+Sample+Purge)
                     +(2.025)*np.pi*0.0025**2 #Chamber->Bomba
                     +(0.015)*np.pi*0.0015**2 #Bomba->Nafion
                     +(0.092)*np.pi*0.0015**2 #Nafion->R7
                     +(0.063)*np.pi*0.0015**2 #R7-Purge
                     +(0.093)*np.pi*0.0015**2+(0.0145)*np.pi*0.0025**2), #Purge->Chamber
            'RS':0}
    # Volumen camaras (m3)
    V_ch = {'Cnt': 0.768*0.567*0.340, # Medidas hasta 18/12/2018: 0.768*0.567*0.342
            'Cn2': 0.767*0.570*0.268, # Medidas hasta 18/12/2018: 0.767*0.570*0.274
            'V10': 0.368*0.269*(0.126-0.026), 'V15': 0.368*0.268*(0.166-0.026),
            'V15ne': 0.368*0.268*(0.166-0.003), 'V35': 0.567*0.37*(0.166-0.024),
            'V35ne': 0.567*0.370*(0.166-0.003), 'VUPC': 0.37*0.267*0.118,
            'VUPCTh': 0.37*0.267*0.118 + 0.002 + 0.18*np.pi*0.002**2,
            'C0': 0.00960, 'C3': 0.00999, 'C6': 0.00946,
            'C9': 0.00957, 'C12': 0.00979,'C14': 0.00968}
    # Superficie camaras
    S_ch = {'Cnt': 0.768*0.567, 'Cn2': 0.767*0.570, 'V10': 0.368*0.269, 
            'V15': 0.368*0.268, 'V15ne': 0.368*0.268, 'V35': 0.567*0.37,
            'V35ne': 0.567*0.370, 'VUPC': 0.37*0.267, 'VUPCTh': 0.37*0.267,
            'C0': 0.0464, 'C3': 0.0466, 'C6': 0.0468, 
            'C9': 0.0465, 'C12': 0.0466,'C14': 0.0465}
    ###########################################################################
           
    ## Exhalacion #############################################################
    # Calculamos el area y el volumen
    if experiment_info[2] in V_ch.keys():
        S = S_ch[experiment_info[2]]
        V = V_ch[experiment_info[2]]+V_ap[experiment_info[1]]
    else:
        chamber_label, height = experiment_info[2].split('-')
        S = S_ch[chamber_label]
        V = V_ch[chamber_label] + S*np.float32(height)/1000 + V_ap[experiment_info[1]]
    # Nos aseguramos de que todo está bien
    assert S>0 and V>0
    # Comprobamos el tipo de ajuste
    if fitType == 'lin':
        # Calculamos la exhalacion (*3600: s -> h)
        E = V/S*popt[0]*3600
        # Calculamos el error
        sE = perr[0]*V/S*3600
        # Guardamos info de los parametros
        m = popt[0]
        sm = perr[0]
        n = popt[1]
        sn = perr[1]   
        # Save in a tuple
        exhalation_results = [E,sE,m,sm,n,sn]
        labels= ['Exhalation (Bq m-2 h-1)','sE','m (Bq m-3 s-1)','sm',
                 'n (Bq m-3)','sn']
    elif fitType == 'par':
        # Calculamos la exhalacion (*3600: s -> h)
        E = V/S*popt[0]*3600
        # Calculamos el error
        sE = perr[0]*V/S*3600
        # Calculamos la Lambda y su error
        Lamb = -2*popt[2]/popt[0]
        sLamb = Lamb*np.sqrt(np.square(perr[2]/popt[2]
                             +np.square(perr[0]/popt[0])))
        # Guardamos info de los parametros
        m = popt[0]
        sm = perr[0]
        n = popt[1]
        sn = perr[1]   
        o = popt[2]
        so = perr[2] 
        # Save in a tuple
        exhalation_results = [E,sE,m,sm,n,sn,o,so,Lamb,sLamb]
        labels= ['Exhalation (Bq m-2 h-1)','sE','m (Bq m-3 s-1)','sm',
                 'n (Bq m-3)','sn','o (Bq2 m-6 s-2)','so',
                 'Lamb (s-1)','sLamb']
    elif fitType == 'exp':
        # Calculamos la exhalacion (*3600: s -> h)
        E = V/S*popt[0]*popt[1]*3600
        # Calculamos el error
        sE = (V/S*np.sqrt(np.square(popt[0]*perr[1])
                            +np.square(popt[1]*perr[0]))*3600)
        # Guardamos info de los parametros
        CSat = popt[0]
        sCSat = perr[0]
        Lamb = popt[1]
        sLamb = perr[1]   
        # Save in a tuple
        exhalation_results = [E,sE,CSat,sCSat,Lamb,sLamb]    
        labels = ['Exhalation (Bq m-2 h-1)','sE','CSat (Bq m-3)','sCSat',
                  'Lamb (s-1)','sLamb']
    elif fitType == 'expGen':
        # Calculamos la exhalacion (*3600: s -> h)
        E = V/S*popt[0]*popt[2]*3600
        # Calculamos el error
        sE = (V/S*np.sqrt(np.square(popt[0]*perr[2])
                            +np.square(popt[2]*perr[0]))*3600)
        # Guardamos info de los parametros
        CSat = popt[0]
        sCSat = perr[0]
        C0 = popt[1]+popt[0]
        sC0 = np.sqrt(perr[1]**2+perr[0]**2)
        Lamb = popt[2]
        sLamb = perr[2]   
        # Save in a tuple
        exhalation_results = [E,sE,C0,sC0,CSat,sCSat,Lamb,sLamb] 
        labels = ['Exhalation (Bq m-2 h-1)','sE','C0 (Bq m-3)','sC0',
                  'CSat (Bq m-3)','sCSat','Lamb (s-1)','sLamb']
    ###########################################################################

    ## Return
    return (exhalation_results,labels)
# =============================================================================

# == Analyze the existing files ===============================================
# We are going to loop over a directory and list all folders with a given name.
# Then we are going to save that information for the next time we run the test,
# we can know which files are already tested and which ones not.
def files_update(dirPath=None,
                 fitType='expGen',t_initial=10,t_duration=None,
                 initial_parameters=None):
    """
    This functions navigates through all subdirectories in the given path and
    computes the given fit.
    
    The function will use a excel file in the directory named 
    "analyzedExperiments.xlsx" which controls the files that have been read 
    until now.
    
    Known Bugs:
        - If some of the files are erased, they will still appear in the file
            "analyzedExperiments.xlsx" but may not appear in data file.
    """
    ## Path handling ##########################################################
    dirPath = open_FileOrDir(dirPath,kind='dir') 
    # Build the path to 'analyzedExperiments.csv'
    filePath = os.path.join(dirPath,'analyzedExperiments.xlsx')
    ###########################################################################
    
    ## Handling analyzedExperiments.csv file ##################################
    # The file will have a column for each fit conditions that includes all 
    # files that has been analyzed using that type of fit

    # Build the fit conditions string
    desired_fit = '{}_{}-{}'.format(fitType,t_initial,t_duration)
    # Check if file exists
    if os.path.exists(filePath):
        # Load information
        analyzedExp_df = pd.read_excel(filePath,sheet_name=None)
        # Check what files have been studied with the desired fit
        if desired_fit in analyzedExp_df.keys():
            # Extract the df of interest
            analyzedExp_df = analyzedExp_df[desired_fit]
            # Extract the list of file names
            analyzedFiles = analyzedExp_df['Filename'].tolist()
            # Extract the list of file paths
            analyzedPaths = analyzedExp_df['Filepath'].tolist()
            # Loop over all files, checking if it is already extracted
            scannedFiles = files_loop(dirPath)
            # Check files in scannedFiles that are not in analyzedFiles 
            newFiles = [file for file in scannedFiles if file[0] not in analyzedFiles]
            
            ## Check if there are erased files
            # Untuple the file names
            scannedFiles_list = [file[0] for file in scannedFiles]
            # Check files in analyzedFiles that are not in scannedFiles
            erasedFiles = [file for file in analyzedFiles
                           if file not in scannedFiles_list]     
        else:
            # use function "files_loop()" to loop over everything
            newFiles = files_loop(dirPath)
            # State that there is no alreadyReadFiles
            analyzedFiles = []
            # State that there is no alreadyReadPaths
            analyzedPaths = []
    # If file does not existes, we need to loop over everything
    else:
        # use function "files_loop()" to loop over everything
        newFiles = files_loop(dirPath)
        # State that there is no alreadyReadFiles
        analyzedFiles = []
        # State that there is no alreadyReadPaths
        analyzedPaths = []
    ###########################################################################
    
    ## Read files #############################################################
    # Initialize the list to store data
    results = []
    # Iterate over scannedFiles
    for file in newFiles:
        # Extract data
        df, expInfo = extract_RadonData(file[1])
        # Only use Closed Circuit Experiments
        if expInfo[4] == 'CC':
            # Compute fit
            fitResults = compute_fit(df,fitType,t_initial,t_duration)
            # Compute exhalation
            exhResults,exhLabels = compute_exhalation(fitResults,expInfo)
            # Save the information obtained
            results.append(expInfo + exhResults + fitResults[3:6])
    ###########################################################################
    
    ## Append new data to old data ############################################
    ## Build the new data list
    if newFiles:
        # Save column names
        labels = (['Date','Device','Chamber','Soil','Method','Placement'] 
                    + exhLabels + ['R2adj','chi2','p-value'])
        # Create the dataframe
        df_new = pd.DataFrame.from_records(results,columns=labels)
    else:
        df_new = pd.DataFrame()
    ## Get the old data DataFrame
    # Get the upstream directory and build the folder name
    exhDataDir = os.path.join(os.path.dirname(dirPath),'cleanData/')
    # Build the dataFile name
    exhDataFile = os.path.join(os.path.join(exhDataDir,'exhData.xlsx'))
    # Check if the excel file is there
    if os.path.exists(exhDataFile):
        # Read the old data
        df_old = pd.read_excel(exhDataFile,sheet_name=None)
        # Check what files have been studied with the desired fit
        if desired_fit in df_old.keys():
            # Extract the df of interest
            df_old = df_old[desired_fit]
            # Combine them
            df_updated = pd.concat([df_old,df_new],axis=0)
        else:
            df_updated = df_new
    else:
        # Build an empty dataframe
        df_updated = df_new
    # Save the excel
    saveToExcel(df_updated,desired_fit,exhDataFile)
    ###########################################################################
    
    ## Update 'analyzedExperiments.excel' #####################################
    # Build the new information dictionary
    updated_files = analyzedFiles + [file[0] for file in newFiles]
    updated_paths = analyzedPaths + [file[1] for file in newFiles]
    # Make sure dimensions are equal
    assert len(updated_files) == len(updated_paths)
    # Build the dataframe to save
    df = pd.DataFrame()
    # Fill it with information
    df['Filename'] = updated_files
    df['Filepath'] = updated_paths
    # Save to excel
    saveToExcel(df,desired_fit,filePath)
    ###########################################################################
    
    # Return the updated df
    return df_updated
    ###########################################################################

# =============================================================================

# == Loop over files ===========================================================================
def files_loop(dirPath=None):
    """
    This function loops over every subdirectory in filePath, 
    if the directory tree looks like:
        
    (Soil)/Camara_(chamber)/(experiment)/(experiment).txt
    
    where the experiment structure is:
        
    '(yyyy)-(mm)-(dd)-(hhmm)_(device)_(chamber)_(soil)_(method)_(placement)'

    it will save the path to the archive into a list, returning it.
    """
    ## Path handling 
    dirPath = open_FileOrDir(dirPath,kind='dir')
    
    ## Loop over files ########################################################
    # Initialize the lists
    readFiles = []
    # Start the loop
    for root, dirs, filesRaw in os.walk(dirPath):
    ## Valid-directory control ##
    # We will extract the folders leading to the file.
    # The path should have the form: '*/(soil)/(chamber)/(experiment)/*'.
    # This happens when:
    # 1. Soil folder is 'Cnt', 'Cn2' or 'BFY'
    # 2. Chamber folder begins with 'Camara'
    # 3. Experiments folder begins with '20'
    ## ##########################
        # Try to check if the conditions are valid    
        try:
            # Slice root folder in its parts
            root_parts = root.split('\\')
            # soil test
            flag_contenedor = root_parts[-3] in ['Cnt','Cn2','BFY']
            # chamber test
            flag_camara = root_parts[-2][:6] == 'Camara'
            # experiment test
            flag_exp = root_parts[-1][:2] == '20'
            # Check if everything is correct
            flag = flag_contenedor and flag_camara and flag_exp
        # If we cannot check, we cannot go on
        except:
            flag = False
        # If flag is True, move on
        if flag:
            # Takes in only txt files
            files = [file for file in filesRaw if file.split('.')[-1] == 'txt']
            # Filter by (device) information
            files = [file for file in files 
                     if file.split('_')[1] in ['aG','aC','aGv2','aCv2','R7v2',
                                                  'R7Sniff','R7DRY','RS']]
            # Check if there is only one file
            if len(files) != 1:
                print('More than one file in: {}'.format(root))
                print('\n Check valid device names.')
            # Take out the item from the list
            file = files[0]
            # Save the file and its path
            readFiles.append((file,os.path.join(root,file)))
    ###########################################################################
    # Return results
    return readFiles
# =============================================================================
        
# == Saving to an excel =======================================================
def saveToExcel(df,excelSheet,excelFile=None):
    """
    This function saves an existing dataframe to an excel. If the excel already 
    exists, the function will ensure that all existing sheets are preserved.
    """
    ## Path handling 
    excelFile = open_FileOrDir(excelFile,kind='file')

    ## Save to excel ##########################################################
    # Create the writer
    writer = pd.ExcelWriter(excelFile, engine='openpyxl')
    # Check if the file already exists
    if os.path.isfile(excelFile):
        # Create the book
        book = load_workbook(excelFile)
        # Associate the book the writer
        writer.book = book
        # Save the existing sheets
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        # Save the dataframe to a specific sheet
        df.to_excel(writer,sheet_name=excelSheet,startrow=0,startcol=0)
        # Save the state of the writer
        writer.save()
    else:
        df.to_excel(writer,sheet_name=excelSheet)
        writer.save()
    ###########################################################################
# =============================================================================

# == Path input ===============================================================
def open_FileOrDir(path=None,kind='file'):
    """
    This function takes in a variable and check if it exists. If no variable
    is provided, it will ask the user to enter a valid option using a window.
    """
    ## Assertions
    assert kind in ['file','dir']
    ## Path handling 
    # Check if a filePath was given
    if path:
#        # Check if the directory exists    
#        assert os.path.exists(path)
        # Save it as outpout
        outputPath = path
    # If there is no path, open a dialog box to find it.
    else:
        # Create the main window and hide it
        root = Tk()
        root.withdraw()
        # Ask for the path
        if kind == 'file':
            outputPath = askopenfilename(title='Select the file:')
        elif kind == 'dir':
            outputPath = askdirectory(title='Select the directory:')    
    return outputPath
# =============================================================================

if __name__ == "__main__":
    print("Doing stuff...")
    
#    df,expInfo = extract_RadonData()
#    # Calculamos el espaciado temporal entre valores 
#    t_step = (df.index.values[1] - df.index.values[0])
#    # Calculamos los segundos que ha durado el experimento
#    t_Raw = df.index.values - df.index.values[0] + t_step
#    # Lo pasamos a segundos
#    t_Raw = t_Raw/np.timedelta64(1,'s')
#    # Cogemos los datos de radon y su error
#    Rn_Raw = np.array(df['Radon'])
#    RnE_Raw = np.array(df['Radon Error'])
#    #parabolic
#    fig = plt.figure()
#    fitResults = compute_fit(df,fitType='par')
#    exhResults, labels = compute_exhalation(fitResults,expInfo)    
#    Exh_par,sExh_par = exhResults[0], exhResults[1]
#    Lamb_par,sLamb_par = exhResults[-2], exhResults[-1]
#    plt.plot(t_Raw,Rn_Raw,label='Data',marker='.',linestyle='none')
#    plt.plot(t_Raw,fitParams['par'][0](t_Raw,*fitResults[1]),
#             label='par')
#    # exponential
#    fitResults = compute_fit(df,fitType='lin',t_duration=30,
#                             initial_parameters=[1e-2,1e8,1e-5])
#    exhResults, labels = compute_exhalation(fitResults,expInfo)
#    Exh_expGen,sExh_expGen = exhResults[0], exhResults[1]
#    Lamb_expGen,sLamb_expGen = exhResults[-2], exhResults[-1]
#    plt.plot(t_Raw,fitParams['expGen'][0](t_Raw,*fitResults[1]),
#             label='expGen')
#    plt.legend()
#    print('Exhalation\nParabolic: {:.2E} +- {:.2E}\nExponential: {:.2E} +- {:.2E}'.format( Exh_par,sExh_par, Exh_expGen,sExh_expGen ))
#
#    print('Lamb\nParabolic: {:.2E} +- {:.2E}\nExponential: {:.2E} +- {:.2E}'.format( Lamb_par,sLamb_par, Lamb_expGen,sLamb_expGen ))
#    print(labels)
#    print(exhResults)
#    files_update()
#    files_update(fitType='lin')

## == Extraer info sobre duracion experimento ==================================
# Inicio una lista
#data = []
## Itero sobre los archivos
#files = files_loop('.')
#for file in files:
#    # extraigo su info
#    df, expData = extract_RadonData(file[1])
#    # Compruebo si es R7
#    if (expData[1] == 'aGv2') & (expData[2] == 'Cnt') & (expData[3] == 'Cnt'):
#        # Calculo la distancia entre el primer dato y el ultimo
#        duration = df.index[-1]-df.index[0]
#        # Lo pasamos a horas
#        duration = duration.seconds/3600
#        # Añado a expData la duracion
#        expData.append(duration)
#        # Añado expdata a la lista final
#        data.append(expData)
#print(data)
## =============================================================================
