# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:50:19 2016

@author: lulin
"""
import random as rd
import numpy as np
import thePlat

#The Manufacturer/The Principal: platform manufacturer and manager initiates with release step (time) and amount
#                 of generations to release [releases]. Contains platforms, games, users, sales. Spreads 
class thePrincipal(object):
    def __init__(s,ID,price,time,releases,features='uniform',release_time=None,massinf=1): 
        s.ID = str(ID)
        s.plat,s.games = [],[]                    #no plat, no plat games
        s.iplat,s.igames = ['0'],['0-0']          #on development
        s.devs = [ID]                             #first party development
        s.users,s.hwsales,s.swsales = [],[],[]    #Userbase data
        s.releases = releases                     #Number of Generations to release
        s.features = features
        s.expect = 0.01

        if release_time == None:
            s.release_d = [rd.randint(10+(150*i),10+time+(150*i+1)) for i in range(releases)]     #Release steps list
        else:
            s.release_d = [release_time*((i+1)) for i in range(releases)]
        
        s.mkt_operation,s.mkt_rec = 0,[]          #Marketing operatino = initial duration, mkt_rec is TS
        s.mkt_range = massinf                     #Percentage of population to pick for each promotion step
        s.consumerprice,s.revenue = price,0       #Fixed price with fixed wealth (50% population may accquire)
        
    def step(s,market):
        if len(s.release_d) > 0:
            if market.steps >= int(s.release_d[0]-52) and s.releases > 0: s.announce(market)    #Platform Announcement (Public Info)
            if market.steps == s.release_d[0] and s.releases > 0: s.release(market)       #Platform Release (Access to plat)
        
        #Promote platform
        s.marketing(market)                                                           #Keep Positive Public Info
        
        s.mkt_rec.append(s.mkt_operation)
        
        
    def marketing(s,market):
        if s.mkt_operation > 0:                                  #Declines all steps.
            s.mkt_operation -= 1                                 #Lose marketing

            if s.mkt_operation < 26+52:
                #Pick PROMOTION VALUE as percentage of cons, increase their platform expectative
                pick_cons = rd.sample(market.cons,int(len(market.cons)*s.mkt_range))
                
                for con in pick_cons:
                    if market.cons[con.ID].adopt['plat'][str(s.plat[-1])] == 0:
                        influence = np.clip(market.cons[con.ID].expect['plat'][str(s.plat[-1])]+s.expect,-1,1)
                        market.cons[con.ID].expect['plat'][str(s.plat[-1])] = influence


                    #if market.steps in range(48,57):
                    #    print s.ID
                    #    print market.cons[con.ID].expect['plat'][str(s.plat[-1])]
                    #elif market.steps in range(105,115):
                    #    print s.ID
                    #    print market.cons[con.ID].expect['plat'][str(s.plat[-1])]
            #c.expect['plat'][str(s.plat[-1])] = rd.uniform(0,1)
        
    def announce(s,market):
        if any(p not in market.announced_plat for p in s.plat) or s.plat == []:
          #  print 'Anouncing!'
            s.mkt_operation += 26+52+52                                   #Mkting + 100     
            s.plat.append(str(s.ID)+'_'+str(len(s.plat)))            #New platform in self record 'ID'
            market.announced_plat.append(str(s.plat[-1]))            #Market record 'ID'
            for c in market.cons:                                    #Initial Marketing 25% prob
                c.experience['plat'][str(s.plat[-1])] = 0
                c.adopt['plat'][str(s.plat[-1])] = 0
                c.expect['plat'][str(s.plat[-1])] = 0
                c.old_expect['plat'][str(s.plat[-1])] = 0
                c.bought['plat'][str(s.plat[-1])] = 0
                c.store['plat'][str(s.plat[-1])] = 0
                if rd.random() < 0.05:
                    c.expect['plat'][str(s.plat[-1])] = c.expect_m+0.1                                         
                #c.expect['plat'][str(s.plat[-1])] = np.clip(np.random.lognormal(0.15,0.15)-1,-1,1)
                
            #Create the platform object in market
            new_plat = thePlat.thePlat(str(s.plat[-1]),s.release_d[0],s.features)
            market.plats.append(new_plat) #Plat list (object(ID),relase day)
            market.plat_features[new_plat.ID] = new_plat.product                #Store platform features in market 
            
            #Initialize market adoption, devs, expect (for data record)
            market.record[new_plat.ID] = [0]*market.steps                   
            market.dev_rec[new_plat.ID] = [0]*market.steps
            market.expec_rec[new_plat.ID] = [0]*market.steps
        
        
    def release(s,market):
        expects = [c.expect['plat'][str(s.plat[-1])] for c in market.cons]
        expects.sort()
        top2 = int(len(expects)*.98)
        innovator_trsh = expects[top2]                     #FIXED INNOVATORS: Those on the top 2% of expectation are innovators
        for c in market.cons:
            if c.expect['plat'][str(s.plat[-1])] > innovator_trsh:
                c.prospect = str(s.plat[-1])
        market.released_plat.append(str(s.plat[-1]))
        del s.release_d[0]
        s.releases -= 1
     #   print '--- Manager ID ('+str(s.ID)+') released:'+str(s.plat[-1])