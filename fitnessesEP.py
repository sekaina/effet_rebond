#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soukaina

"""
import pandas as pd
import numpy as np
import time

from eppy.modeleditor import IDF
from energyplus_wrapper import EPlusRunner
import config

import multiprocessing as mp

IDDPATH = config.IDDPATH
EPLUSPATH = config.EPLUSPATH

IDFPATH = "./model/"
LIBWINDOW = "./model/windows.idf"

EPWFILE = IDFPATH + "CHAMBERY.epw"


def initialize(idf):
    print("Eppy initialization")
    IDF.setiddname(IDDPATH)
    model = IDF(idf, EPWFILE)
    return model

def build_library(LIBWINDOW, EPWFILE):
    config.IDF_WINDOWS = IDF(LIBWINDOW, EPWFILE)
    for glazing in config.IDF_WINDOWS.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"]:
        config.IDF_WINDOWS.newidfobject(
            "CONSTRUCTION", Name="window_" + str(glazing.Name), Outside_Layer=glazing.Name)

def modify(ind, model):
    print("modifying model %s with ind %s" % (model.idfname, ind))
    surface_mat = []

    surface_mat.append(modify_thickness_insulation_wall(ind[0], model))
    surface_mat.append(modify_thickness_insulation_ceiling(ind[1], model))
    surface_mat.append(modify_thickness_insulation_floor(ind[2], model))
    surface_mat.append(modify_window(ind[3], model))
    #model.saveas("modelIDM"+str(ind[0])+'_'+str(ind[1])+'_'+str(ind[2])+'_'+str(ind[3])+".idf")
    print("surface_mat %s " % (surface_mat))
    return model, surface_mat  

def modify_window(ind_window, model):
    print("modify windows")
    area = 0
    window_ind = config.IDF_WINDOWS.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"][ind_window]
    window_name=window_ind.Name
    windows_construction=["pf_S_R+1","pf_S_RDC","pf_O_RDC","f_E_R+1","f_R+2_E-RDC","f_O_RDC","f_N_RDC","f_N_R+1","pf_O_R+1"]
    windows_shades=["VR_pf_S_R+1","VR_pf_S_RDC","VR_pf_O_RDC","VR_f_E_R+1","VR_f_R+2_E-RDC","VR_f_O_RDC","VR_f_N_RDC","VR_f_N_R+1","VR_pf_O_R+1"]#F_R+2_E-RDC
    model.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"].append(window_ind)
    for window in model.idfobjects["CONSTRUCTION"]:
        if window.Name in windows_construction:
            window.Outside_Layer=window_name
        if window.Name in windows_shades:
            window.Layer_2=window_name
    windows=[object for object in model.idfobjects["FENESTRATIONSURFACE:DETAILED"] if object.Surface_Type=="Window"]
    for window in windows:
        area += window.area

    return area, window_name, ind_window

def modify_thickness_insulation_ceiling(ep, model):
    print("modify ceiling")
    ep_metre=ep*0.01
    area = 0
    construction_Name = "PB_COMBLES_isole"
    materials=model.idfobjects["Material"]
    for material in materials:
        if material.Name=="LDV35_40cm":
            material.Thickness=ep_metre
    for surface in model.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if (surface.Construction_Name == "PB_COMBLES_isole"):
            area += surface.area
    return area, construction_Name, ep

def modify_thickness_insulation_floor(ep, model):
    print("modify floor")
    ep_metre=ep*0.01
    area = 0
    construction_Name = "PB_RDC_isole"
    materials=model.idfobjects["Material"]
    for material in materials:
        if material.Name=="PolystyreneXtrude30":
            material.Thickness=ep_metre
    for surface in model.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if (surface.Construction_Name == construction_Name):
            area += surface.area
    return area, construction_Name, ep

def modify_thickness_insulation_wall(ep, model):
    print("modify walls")
    ep_metre=ep*0.01
    area = 0
    construction_Name = "MurExt_isole"
    materials=model.idfobjects["Material"]
    for material in materials:    
        if material.Name=="LDV35_20_MurExt":
            material.Thickness=ep_metre
    for surface in model.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if (surface.Construction_Name == construction_Name):
            area += surface.area

    return area, construction_Name, ep

def heating_needs(result):
    print("computing heating needs")
    import csv
    for data in result:
        if "table" in data:
            print("found table")
            table=result['table']
            #print(table[49])
            Chauffage = float(table[49][13])# [49][6] in EP v9.3   [49][13] in EP v9.5 pour simulation annuelle
    return Chauffage

def evaluate_model(model, indb, surfacemat):
    start_time = time.time()
    print("running energyplus")
    runner = EPlusRunner(EPLUSPATH)

    simulation = runner.run_one(model, EPWFILE) #extra_file dans le cas de fmu NoMASS
    result=simulation.time_series

    print("Evaluation of %s with %s in %s s" %
                     (model.idfname, indb, time.time() - start_time))

    print("Computing objectives from result dataframes")
    heating = float(heating_needs(result))
    heating_m2=heating/config.building_area
    print("In table.csv, %s kWh, %s kWh/m2" % (heating, heating_m2)) # kwh par m2
    comfort = float(overheating(result))

    print("%s hours of discomfort (where temperature is above Tconf+2°C) " % (comfort))

    print("computing investment price")
    investment = np.array(economy_investment(surfacemat)).sum()
    print("Investment Price %s " % (investment))
    print("computing operation price")
    operation = economy_operation(heating)
    print("Operation Price %s " % (operation))
    total_price = investment + operation
    total_price_m2 = total_price/config.building_area # euros par m2
    print("Total Price %s euros/m2" % (total_price_m2))
    return heating_m2, comfort, total_price_m2


def evaluate(ind):
    surfacemat = []
    
    epmodel = initialize(config.building)
    print(config.building)
    build_library(LIBWINDOW, EPWFILE)
    print("modifying model %s" % (epmodel.idfname))
    epmodel, surfacemat = modify(
        ind, epmodel)
    print("launching evaluation function")
    fitness = np.array(evaluate_model(epmodel, ind, surfacemat))
    print("fitness for %s : %s" % (epmodel.idfname, fitness))
    print("returning fitness")

    return fitness

def economy_investment(surface_mat):
    material = {"Polystyrene": "polystyrene_price",
                #"Rockwool": "rockwool_price",
                "Glasswool": "glasswool_price",
                #"Polyurethane": "polystyrene_price",
                "Window": "window_price"
                }
    price_mat = []
    
    for paroi in surface_mat[:-1]:
        e=paroi[2]
        #try:
        price_this_mat = 0
        if (paroi[1]=="MurExt_isole" or paroi[1]=="PB_COMBLES_isole"):
                mat="Glasswool"
        elif paroi[1]=="PB_RDC_isole":
                mat = "Polystyrene"
        price_this_mat = globals()[material[mat]](e)
        price_mat.append(paroi[0] * (price_this_mat + 10))  # +10 for coating
            
        #except AttributeError:
        #    logger.error("eco : No price for %s" % (mat))
        #    pass
    window=surface_mat[-1]
    price_window = window_price(window)
    price_mat.append(price_window)
    return price_mat


def window_price(window):
    #cout de la forme a*surface_fenetre+b
    a = {0: 460.45, 1: 454.16, 2: 390.85, 3:349.35 }
    b= {0: 34.45, 1: 36.62, 2: 29.37, 3: 28.17}

    price = a[window[2]]*window[0]+b[window[2]]

    #price = price + 49  # 49€ is flat price for arranging walls for construction

    return price

def glasswool_price(e):
    return 0.39 * e + 0.17 # price_this_mat

def polystyrene_price(e):
    return 1.25 * e + 1

def moyenne_glissante(valeurs, intervalle):
    indice_debut = (intervalle - 1) // 2
    liste_moyennes=valeurs[:intervalle-1]
    liste_moyennes += [sum(valeurs[i - indice_debut:i + indice_debut + 1]) / intervalle for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes
def moyenne_glissante_norme (valeurs, intervalle):
    indice_debut=(intervalle - 1) // 2
    liste_moyennes=valeurs[1:intervalle]
    liste_moyennes += [(0.2*valeurs[i - indice_debut]+0.3*valeurs[i - indice_debut+1]+0.4*valeurs[i - indice_debut+2]+
                    0.5*valeurs[i - indice_debut+3]+0.6*valeurs[i - indice_debut+4]+0.8*valeurs[i - indice_debut+5]+
                    valeurs[i - indice_debut+6]) / 3.8 for i in range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes
def overheating(result):
    print("computing overheating")
    indoor = None
    out = None
    heures_inconfort=[]
    oh = []
    for data in result:
            if "eplus" in data:
                print("found eplus.csv")
                indoor = result[data].iloc[:, [
                    "Mean Air Temperature" in col for col in result[data].columns]]
                out = result[data].iloc[:,[
                    "Outdoor Air Drybulb Temperature" in col for col in result[data].columns]]
                Text_moy_jour=[float(out[i:289+i].mean()) for i in range(0,len(out),288)]
                Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)#moyenne glissante sur 7 jours selon la norme NF EN 16798-1
                Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes] # temperature de confort adaptatif selon la norme NF EN 16798-1
                for zone, area in config.zones_areas.items():
                    oh_zone=0
                    heures_inconfort_zone=0
                    indoor_zone=indoor.iloc[:,[zone in col for col in indoor.columns]]
                    T_moy_jour=[float(indoor_zone[i:289+i].mean()) for i in range(0,len(indoor_zone),288)]
                    for i in range(len(T_moy_jour)):
                        if T_moy_jour[i]>(Tconfort[i]+2):
                            oh_zone+=T_moy_jour[i]-(Tconfort[i]+2)
                            heures_inconfort_zone+=1
                    oh.append(oh_zone)
                    heures_inconfort.append(heures_inconfort_zone)
    area_tot=config.building_area
    areas=[]
    for zone,area in config.zones_areas.items():
        areas.append(area)
    oh_tot=sum([x*y for x,y in zip(areas,oh)])/area_tot  #somme pondérée par les surfaces
    heures_inconfort_tot=sum([x*y for x,y in zip(areas,heures_inconfort)])/area_tot  
    print("overheating = %s °C/h" % (oh_tot))
    print("heures inconfort = %s " % (heures_inconfort_tot))
    return heures_inconfort_tot

def overheating_from_csv(csv_filename):
    print("computing overheating")
    indoor = None
    out = None
    heures_inconfort=[]
    oh = []
    data=pd.read_csv(csv_filename)
    indoor = data.iloc[:, [
        "Mean Air Temperature" in col for col in data.columns]]
    out = data.iloc[:,[
        "Outdoor Air Drybulb Temperature" in col for col in data.columns]]
    Text_moy_jour=[float(out[i:289+i].mean()) for i in range(0,len(out),288)]
    Text_glissantes=moyenne_glissante_norme(Text_moy_jour, 7)#moyenne glissante sur 7 jours selon la norme NF EN 16798-1
    Tconfort=[0.33*Tmoyext+18.8 for Tmoyext in Text_glissantes] # temperature de confort adaptatif selon la norme NF EN 16798-1
    for zone, area in config.zones_areas.items():
        oh_zone=0
        heures_inconfort_zone=0
        indoor_zone=indoor.iloc[:,[zone in col for col in indoor.columns]]
        T_moy_jour=[float(indoor_zone[i:289+i].mean()) for i in range(0,len(indoor_zone),288)]
        for i in range(len(T_moy_jour)):
            if T_moy_jour[i]>(Tconfort[i]+2):
                oh_zone+=T_moy_jour[i]-(Tconfort[i]+2)
                heures_inconfort_zone+=1
        oh.append(oh_zone)
        heures_inconfort.append(heures_inconfort_zone)
    area_tot=config.building_area
    areas=[]
    for zone,area in config.zones_areas.items():
        areas.append(area)
    oh_tot=sum([x*y for x,y in zip(areas,oh)])/area_tot  #somme pondérée par les surfaces
    heures_inconfort_tot=sum([x*y for x,y in zip(areas,heures_inconfort)])/area_tot  
    print("overheating = %s °C/h" % (oh_tot))
    print("heures inconfort = %s " % (heures_inconfort_tot))
    return heures_inconfort_tot


def economy_operation(Echauffage):
    cost=config.K*config.Pelec*(Echauffage/config.COP) # PAC de COP=4
    return cost

def modify_consigne(T, model):
    print("modify heating setpoint")
    thermostat=model.idfobjects["HVACTEMPLATE:THERMOSTAT"]
    print(thermostat)
    for x in thermostat:
        print (x.Heating_Setpoint_Schedule_Name)
        x.Heating_Setpoint_Schedule_Name='On'
        x.Constant_Heating_Setpoint=T
    model.saveas("./model/modelIDM"+str(T)+".idf")
def modify_Schedule_File(model,i):
    thermostat=model.idfobjects["SCHEDULE:FILE"]
    #print(thermostat)
    for x in thermostat:
        #print (x)
        x.File_Name="Tin_"+str(i)+".csv"
    model.saveas("./model/modelIDM_menage"+str(i)+".idf")
    return model
def evaluate_rebound(T):
    epmodel = initialize(config.building)
    print("modifying setpoint %s" % (epmodel.idfname))
    model_modified= modify_consigne(T, epmodel)

    #return fitness

def model_kelly(Text,T_stat,SettingResp,TRV,CH_Hours,Reg_Pat,Auto_Timer,HH_Size,HH_Income,Child_5, Child_18, Age_60, Age_60_64, Age_64_74, Age_74, Owner, Renter,WE_Same,WE_Temp,Elec_Main, Build_Age, Roof_Ins,Dbl_Glz, Wall_U):
    x_Text=0.052
    x_Text2=0.012
    x_T_stat=-0.236
    x_SettingResp=0.035
    x_TRV=-0.169
    x_CH_Hours=0.069
    x_Reg_Pat=1.189
    x_Auto_Timer=-0.031
    x_HH_Size=0.25
    x_HH_Income=0.084
    x_Child_5=0.495
    x_Child_18= 0.219
    x_Age_60= 0
    x_Age_60_64= 0.051
    x_Age_64_74=0.37
    x_Age_74=0.585
    x_Owner=0
    x_Renter=0.94
    x_WE_Same=-0.438
    x_WE_Temp=0.038
    x_Elec_Main=-0.195
    x_Build_Age=0.042
    x_Roof_Ins=0.125
    x_Dbl_Glz=0.188
    x_Wall_U=0.076
    Alpha=14.224
    Tconsigne=Alpha+x_Text*Text+x_Text2*(Text**2)+x_T_stat*T_stat+x_SettingResp*SettingResp+x_TRV*TRV+x_CH_Hours*CH_Hours+x_Reg_Pat*Reg_Pat+\
                x_Auto_Timer*Auto_Timer+x_HH_Size*HH_Size+x_HH_Income*HH_Income+x_Child_5*Child_5+\
                    x_Child_18*Child_18+x_Age_60*Age_60+x_Age_60_64*Age_60_64+x_Age_64_74*Age_64_74+x_Age_74*Age_74+\
                        x_Owner*Owner+x_Renter*Renter+x_WE_Same*WE_Same+x_WE_Temp*WE_Temp+x_Elec_Main*Elec_Main+\
                            x_Build_Age*Build_Age+x_Roof_Ins*Roof_Ins+x_Dbl_Glz*Dbl_Glz+x_Wall_U*Wall_U
    return Tconsigne
def Tin_diff_occupants(N):
    data=pd.read_csv("./model/IDM.csv")
    out = data.iloc[:,["Outdoor Air Drybulb Temperature" in col for col in data.columns]]
    Text_moy_jour=[float(out[i:289+i].mean()) for i in range(0,len(out),288)]
    Text_moy_jour_annee=[]
    for i in range (365):
        Text_moy_jour_annee+=[Text_moy_jour[i]]*24
    #print(len(Text_moy_jour_annee))
    caract=pd.DataFrame()
    for i in range(N):
        Tin=pd.DataFrame()
        Text = 9.71 # a scale variable representing the mean external temperature for a particular region of England. 9.71°C is the mean
        T_stat = 1 # room thermostat exists 
        SettingResp=np.random.randint(0,4) #T_set = the respondent’s declared thermostat setting for the dwelling in degrees Celsius and has been grouped into four categories : 0 if setpoint<18 , 1 if 18<setpoint<20, 2 if 20<setpoint<22, 3 if setpoint>22 
        TRV=0 # 1 if the only type of temperature control is with thermostatic radiator valves
        CH_Hours=np.random.randint(1,25) #a continuous scale variable indicating the average number of central heating hours reported per day over the week including weekends.
        Reg_Pat=1 #a dichotomous variable indicating if the home is heated to regular heating patterns during the winter.
        Auto_Timer=0 #a dichotomous variable indicating that the home uses an automatic timer to control heating.
        HH_Size=np.random.randint(1,8) #the number of occupants living in the dwelling at the time of the survey
        HH_Income= np.random.randint(0,7)#the gross take-home income for the whole household and has been categorised into seven income bands.
        Child_5=np.random.randint(0,2) #a dichotomous variable indicating if any infants under the age of five are present in the dwelling.
        Child_18= np.random.randint(0,5)#a discrete scale variable indicating the number of children under the age of 18 living in the dwelling.
        Age_60=1 #a dichotomous variable indicating if the oldest person living in the dwelling is under 64 years of age. comparison category 
        Age_60_64= np.random.randint(0,2)#a dichotomous variable that represents if the oldest person living in the dwelling is aged between 59 and 64.
        Age_64_74= np.random.randint(0,2)#a dichotomous variable that represents if the oldest person living in the dwelling is aged between 64 and 74.
        Age_74= np.random.randint(0,2) # a dichotomous variable that represents if the oldest person in the dwelling is over 74.
        if (Age_60_64==1):
            Age_64_74=0
            Age_74=0
        if (Age_64_74==1):
            Age_60_64=0
            Age_74=0
        if (Age_74==1):
            Age_60_64=0
            Age_64_74=0
        Owner= np.random.randint(0,2)
        Renter = 1-Owner
        WE_Same = 0#np.random.randint(0,2) #a dichotomous variable indicates a positive response to the question: ‘‘Do you heat your home the same on the weekend as during the week?’’
        WE_Temp=0# np.random.randint(0,2)#a dichotomous variable indicates a weekend temperature recording.
        Elec_Main=1
        Build_Age= 9#np.random.randint(0,10) #10 catégorie : 9= 2002-2006
        Roof_Ins=6#np.random.randint(0,8) # 8 catégories : 6 = 150-200 mm
        Dbl_Glz= 4#np.random.randint(0,5) # 5 catégorie : 4 = all windows
        Wall_U=3#np.random.randint(0,4) # 4 catégories : 3 = <0.4 W/m2.K
        caract=caract.append({'SettingResp' : SettingResp , 'CH_Hours' : CH_Hours, 'HH_Size' : HH_Size, 'HH_Income': HH_Income,\
                                'Child_5': Child_5,'Child_18': Child_18, 'Age_60_64': Age_60_64, 'Age_64_74': Age_64_74, \
                                'Age_74': Age_74, 'Owner': Owner, 'WE_Same': WE_Same, 'WE_Temp':WE_Temp,'Build_Age': Build_Age,\
                                'Roof_Ins': Roof_Ins, 'Dbl_Glz': Dbl_Glz, 'Wall_U': Wall_U} , ignore_index=True)
        consigne=[]
        for t in Text_moy_jour_annee:
            c=model_kelly(t,T_stat,SettingResp,TRV,CH_Hours,Reg_Pat,Auto_Timer,HH_Size,HH_Income,Child_5,\
                        Child_18, Age_60, Age_60_64, Age_64_74, Age_74, Owner, Renter,WE_Same,WE_Temp,\
                        Elec_Main, Build_Age,Roof_Ins,Dbl_Glz, Wall_U)
            consigne.append(c)
        Tin["menage"+str(i)]=consigne
        Tin.to_csv("./model/Tin_"+str(i)+".csv", index=False)
    print (caract.describe())
    caract.to_csv("./model/caract.csv", index=False)
    #fig,axes=plt.subplots()
    '''Tin.plot()
    plt.xlabel("jour")
    plt.ylabel("T [°C]")
    plt.axvline(x = 105, color = 'r', linestyle = '-')
    plt.axvline(x = 289, color = 'r', linestyle = '-')
    plt.fill([105,289,289,105], [0,0,30,30],c='gray')
    plt.legend(fontsize=7)
    plt.show()'''
    return caract
def evaluate_modified_model(model_modified,col):
    print("launching evaluation function")
    start_time = time.time()
    print("running energyplus")
    runner = EPlusRunner(EPLUSPATH)
    extra=["./model/Tin_"+str(col)+".csv"]
    simulation = runner.run_one(model_modified, EPWFILE, extra_files=extra) #extra_file dans le cas de fmu NoMASS
    result=simulation.time_series

    print("Evaluation of %s with %s in %s s" % (model_modified.idfname, col, time.time() - start_time))

    print("Computing objectives from result dataframes")
    heating = float(heating_needs(result))
    heating_m2=heating/config.building_area
    print("In table.csv, %s kWh, %s kWh/m2" % (heating, heating_m2)) # kwh par m2
    return heating
def evaluate2 (i):
    epmodel = initialize(config.building)
    print("modifying column number in Schedule:File %s" % (epmodel.idfname))
    model_modified= modify_Schedule_File(epmodel,i)
    result=evaluate_modified_model(model_modified,i)
    return result
if __name__ == "__main__": #pour tester
    #T=20
    #evaluate_rebound(T)
    N=10
    caract=Tin_diff_occupants(N)
    col=list(np.arange(0,N))
    print(col)
    lock = mp.Lock() 
    start_time = time.time()
    with mp.Pool() as p:
        futures = p.map_async(evaluate2, col)
        print(f"futures: {futures}")
        print(f"futures.get: {futures.get()}")
        with open("monitoring.txt", "w") as f:
            f.write(str(futures.get()))
    # Print running time
    print("--- %s seconds ---" % (time.time() - start_time))
    caract["heating"]=futures.get()
    caract.to_csv("caract_with_heating_needs.csv", index=False)
    
    

