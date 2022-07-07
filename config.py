#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 02 11:03:32 2021

@author: 
"""

indiv = []

building = "./model/IDM.idf" #"./modelNoMASS/IDM_NoMASS.idf"

zones_areas={"RDC THERMAL ZONE":48.75,"ZCH1":14.86,"ZCH2":10.16,"ZCH3":9.64,"ZSDB":14.09}
building_area=97.5
IDF_WINDOWS = None
IDF_WALLS = None


IDDPATH = "C:\EnergyPlusV9-5-0\Energy+.idd"# "/usr/local/EnergyPlus-8-6-0/Energy+.idd"
EPLUSPATH = "C:\EnergyPlusV9-5-0"
TIMESTEP = 12


# Bounds walls
LOW_WALLS, UP_WALLS = 10, 50 # épaisseurs en cm 
# Bounds windows
LOW_WINDOW, UP_WINDOW = 0, 3 # 4 types de fenetres
BOUNDS = [(LOW_WALLS, UP_WALLS), (LOW_WALLS, UP_WALLS),
          (LOW_WALLS, UP_WALLS), (LOW_WINDOW, UP_WINDOW)]



#paramètres pour le calcul du coût d'exploitation (énergie)
inflation=0.01 # inflation
t=0.04 #taux d'actualisation
r=(1+inflation)/(1+t)
N=20 #durée de vie du système en nombre d'années
K=r*(r**N-1)/(r-1)
Pelec=0.15 # prix de l'électricité en euro/kWh
Pgaz= 0.069 #prix gaz en euro/kwh (tarrif Engie)
COP=4

lock = None
