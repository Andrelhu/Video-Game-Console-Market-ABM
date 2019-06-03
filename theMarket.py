    # -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:22:59 2016

@author: lulin
"""
import random as rd
import numpy as np
import thePrincipal
import theDev
import theCons
import pandas as pd

from numba import autojit

#Market object: contains platforms [#,generations], consumers [#,socialdegrees], devs[#,perception],
#               manufacturers(principal) [#]. Tracks [consumer adoption, expectatives/expectation,
#               developers by platform].
class Market(object):
    def __init__(s,platq,devq,consq,price,netd,time,devbound,releases,quality,tot_step,release_dates=None,socinf=0.1,massinf=1,preferences='uniform',features='uniform',console_features='uniform',pref_n=2,feat_n=2): 
        
        #v3.0 : Chapter 3 : Preference and Feature definition
        
        #v3.0 : Multihoming, exclusivity
        s.exclusivity = [0,0]
        perception = 3
        print 'Consumer perception : '+str(perception)
        print 'Targeted developers : '+str(devq/2)

        #Uniforms
        preferences = 'uniform'
        features = 'uniform'
        console_features = 'uniform'
        
        #Sim1 #Done     
 #       typeA = [1,1,1,1,1,-1,-1,-1,-1,-1]
 #       typeB = [-1,-1,-1,-1,-1,1,1,1,1,1]
 #       preferences = [typeA]*(consq/2) + [typeB]*(consq/2)
        
        #Sim2 #Done
   #     typeA = [1,-1]+[1]*8
   #     typeB = [-1,1]+[-1]*8
   #     preferences = [typeA]*(consq/2) + [typeB]*(consq/2)
        
        #Sim3     Features = [typeA]*(consq/2) + [typeB]*(consq/2)
        #typeA = [1,1,1,1,1,-1,-1,-1,-1,-1]
        #typeB = [-1,-1,-1,-1,-1,1,1,1,1,1]
        #features = [[iii*(pref_n-2) for iii in typeA]] * (devq)        
        #features = [typeA] * (devq)
        #console_features = [[iii*(pref_n-2) for iii in typeA]] * (platq)        
        
        #Sim4 
        typeA = [1]*10
   #     typeB = [-1]*10
        preferences = [typeA] * (consq)
        #features = [typeA] * (devq/2) + [typeB] * (devq/2)
   #     preferences = [typeA] * (consq)
        typeB = [1,-1,1,-1,1,-1,1,-1,1,-1]
    #    features = [typeB] * (devq/2) + [typeA] * (devq/2)
        preferences = [typeA]*(consq/2) + [typeB]*(consq/2)
        #console_features = [[iii*(pref_n-2) for iii in typeA]] * (platq)        

        
        #Sim5 and Sim6 #threeplats
        #typeA = [1]*10
        #typeB = [-1]*10
        
        #[[0]*10,typeA,[0]*10]
        #preferences = [typeA]*(consq/2) + [typeB]*(consq/2)
        
        #Sim7+8 (without platform product features)
        typeA = [1]*3  + [-1]*6 + [0]
        typeB = [-1]*3 + [-1]*3 + [1]*3 + [0]
        typeC = [-1]*6 + [1]*3  + [0]
        
        typeGen = [-1]*10
        features = [typeGen] * (devq/2) + [[1,1]+[rd.uniform(-1,1) for i in range(8)]] * (devq/2)
        #features = 'uniform' #[typeA] + [[0]*10] * (devq-1)
        preferences = [typeA]*(consq/3) + [typeB]*(consq/3) + [typeC]*(consq/3)
        if len(preferences) < consq:
            preferences = preferences + [typeC]*(consq-len(preferences))
        

  #      if platq == 2:
  #          console_features = ['uniform',[1]*10] #'uniform'
  #      elif platq == 3:
  #          console_features = ['uniform','uniform',[1]*10] #'uniform'
#        console_features = [[1] * feat_n + [rd.uniform(-1,1) for i in range(10-feat_n)]] * platq
        #features = iter(features)


       
        #v3.0 : Chapter 3 - Features 
        
        #Get iterables
        prices = iter(price)
        if release_dates == None:
            release_dates = iter([None for i in range(platq)])
        else:
            release_dates = iter(release_dates)
            
        
        
        #First initiate platforms
        if console_features == 'uniform':
            s.manuf = [thePrincipal.thePrincipal(i,prices.next(),time,releases,features='uniform',release_time=release_dates.next(),massinf=massinf) for i in range(platq)]
        else:
          #  console_features = [[1]*10]+[[-1]*10]*(platq-1)
           # print console_features
            console_features = iter(console_features)
            s.manuf = [thePrincipal.thePrincipal(i,prices.next(),time,releases,features=console_features.next(),release_time=release_dates.next(),massinf=massinf) for i in range(platq)]
      
        if features == 'uniform':
            s.devs = [theDev.theDev(i,devbound,massinf=massinf,feat_type=features,feat_n=feat_n) for i in range(devq)]#All devs have an ID number, this list considers plats [0 and 1]
        else:
            #features = [typeA] * (devq)
            s.devs = [theDev.theDev(i,devbound,massinf=massinf,feat_type=features[i]) for i in range(devq)]#All devs have an ID number, this list considers plats [0 and 1]

        s.licensed = [dev.ID for dev in s.devs]#[:10]
        
        s.netlinks = 0
        
        #v3.0 : Chapter 3 - Preference 
        if preferences == 'uniform':
            s.cons = [theCons.theCons(i,netd,socinf,preference=preferences,pref_n=pref_n,perception=perception) for i in range(consq)]
        else:
            #preferences = [typeA]*(consq/2) + [typeB]*(consq/2)
            s.cons = [theCons.theCons(i,netd,socinf,preference=preferences[i],perception=perception) for i in range(consq)]
        s.steps = 0
        s.tot_step = tot_step
        s.maxtech = 1
        
        s.plats = []
        s.plat_features = {}
        s.games = {}
        s.games_features = {}
        
        s.pos_ID = {}
        
        #Keep track of Main Behaviors
        s.experiment = {'adopters':{},'cumulative':{},'developers':{},
                        'totals':{},'expectatives':{},'dev_rank':{},
                        'weekly_mean':[],'weekly_std':[]}
        
        s.announced_plat = []                            #Announced platforms
        s.released_plat = []                             #Released platforms
        
        s.retired_plat = []
        s.names = []                                     #???Names of platforms?
        s.record = {str(i):[0] for i in range(platq)}    #Adoption by platform
        s.plat_adopter = {str(i):[0] for i in range(platq)}
        s.dev_rec = {str(i):[1] for i in range(platq)}   #Developers by platform
        s.expec_rec = {str(i):[1] for i in range(platq)} #Expect. by platform
        s.expec_rec['M'] = [0]
        s.expec_rec['M_SD'] = [0]
        s.dev_multihome = [0]
        s.dev_mh_std = [0]
        s.dev_plats = [0]
        s.cm_rec = [0]
        
        #Debugging
        s.debug_expect = {i:{} for i in range(45,65)}
        s.disadoptersnet = 0
        s.disadoptersall = []
        
    def init_agents(s,market):
        [cons.init_me(market) for cons in s.cons]              #Activate consumers
        
    def pos_to_ID(s,market):
        temp_ = {s.cons[i].ID: i for i in range(len(s.cons))}
        #s.pos_ID = pd.DataFrame.from_dict(temp_,orient='index')
        s.pos_ID = temp_
    
    #@autojit    
    def step(s,mkt):                                         #Run one cycle
        #rd.shuffle(s.manuf)
        #rd.shuffle(s.devs)
        #rd.shuffle(s.cons)
        activate = rd.sample(s.cons,int(float(len(s.cons))*0.5))
        
        if s.steps % 2 == 0:
            s.keep_track(mkt)
        #s.pos_to_ID(mkt)

        for cons in activate:
            cons.step(mkt)
        
        for cons in activate:
            cons.update(mkt)
   #     print 'c'
        
        [manf.step(mkt) for manf in s.manuf]
   #     print 'm'
        total_mkt_power = sum([sum([v for v in d.unit_games.values()]) for d in s.devs])
 
        for dev in s.devs:
            dev.mkt_power = float(sum([v for v in dev.unit_games.values()]))/(total_mkt_power+0.01)
       
        [devs.step(mkt) for devs in s.devs]
   #     print 'd' 
        [plat.step(mkt) for plat in s.plats]
   #     print 'p' 
        s.steps += 1
                
    def keep_track(s,market):                                   #Record adoption, devs and expectations
        
        #s.disadoptersall.append(s.disadoptersnet)
        
        
        s.pos_to_ID(market)
    #technology evolution, its fixed and doesnt conttribute tto anything/
        tech = 1 #[manuf.release_d[0] for manuf in s.manuf]
        s.maxtech = 1 #max(tech)

        tp1,tp2,tp3,tp4=[],[],[],[]
        #Consumer loop
                
        for plat in s.plats:
            tp1,tp2,tp3,tp4=[],[],[],[]
            for c in s.cons:
                tp1.append(c.adopt['plat'][str(plat.ID)])
                tp2.append(c.expect['plat'][str(plat.ID)])
                tp3.append(c.expect_m)
                if c.consumeme == True:
                    tp4.append(1)    
            
        #Keep records of:
            
            #Average expectative
            s.expec_rec[str(plat.ID)].append(np.median(tp2))

                
            #Total consumer adopters
            s.record[str(plat.ID)].append(sum(tp1))
            
            #Total developer/publisher adopters
            s.dev_rec[str(plat.ID)].append(len(set([d for d in plat.devs])))
            
            #Mean expectative
            s.expec_rec['M'].append(np.mean(tp3))
            s.expec_rec['M_SD'].append(np.std(tp3))
    
            s.cm_rec.append(sum(tp4))
        
        #Dev loop
        tp1,tp2 = [],[]
        for dev in s.devs:
            tp1.append(len(dev.plats))
            if dev.plats != []:
                tp2.append(1)

        s.dev_multihome.append(np.average(tp1))
        s.dev_mh_std.append(np.std(tp1))
        s.dev_plats.append(len(tp2))