# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:48:43 2016

@author: lulin
"""
import random as rd
#The Platform: as product and service. Created by thePrincipal/theManufacturer.        
class thePlat(object):
    def __init__(s,ID,time,features='uniform'):
        s.ID,s.games,s.announced_games = str(ID),{},[]   #no plat, no plat games
        s.iplat,s.igames,s.devs = ['0'],['0-0'],[ID]     #on development
        s.users,s.hwsales_step,s.hwsales,s.total_licenses = 0,0,[0],9999999
        s.release_d = rd.randint(10,10+time)
        s.release_d2 = rd.randint(150,150+time)
        s.mkt_operation = False
        s.consumerprice = 0.5
        if features == 'uniform':
            s.product = [rd.random(),rd.random()]+[rd.uniform(-1,1) for i in range(8)]
        else:
            s.product = features
        s.netsales = []
        
    def step(s,market):
        s.hwsales.append(s.hwsales_step)
        s.netsales.append(s.hwsales[-1]-s.hwsales[-2])
        if len(s.hwsales) > 48:
            if (s.hwsales[-1]-s.hwsales[-2]) == 0 and len(s.hwsales) > 10400:
                market.retired_plat.append(s.ID)
            if sum(s.netsales[-15:-1]) < sum(s.netsales[-20:-16]):
                market.retired_plat.append(s.ID)