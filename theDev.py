# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:51:10 2016

@author: lulin
"""
#Issue: modeling discussion: 
#       Developers have desincentives when their products compete with too many developers or have relative lower shelf/mind/market share.
#       Way 1) to address this is having #developers as a variable in developers decision making such as 
#           Platform_value = User_population/Dev_population. Aparently this makes developers to optimize and split in almost an even plat adoption
#       Way 2) developers only account for potential user base and restrictions on the platform (licences given by the manufacturer) are the ones
#           that control platform adoption. Giving as second best option the adoption of the second best platform.
#           This way only makes sense if the manufacturer is assumed to have higher bargain power.

import random as rd
import numpy as np

default_dev_time = 52

class theDev(object):
    #initialization of agent parameters and memory
    def __init__(s,ID,bound=0.1,massinf=1.,feat_type='uniform',feat_n=2):
        s.ID = ID                          #ID
        #s.dev_time = 0                     #Game development steps - Original model
        s.dev_time = {}
        s.bound = bound                    #Market sampling boundary
        s.feat_type = feat_type
        s.feat_n = feat_n
        
        #record keeping
        s.plats,s.games = [],0
        s.revenue = 0
    
        s.unit_games = {}
        s.unit_games_ts = {}
        s.unit_games_time = {}
        s.unit_post10weeks = {}
        s.old_adoption = []
        
        #working variables
        s.license_lim = 3               #license restrictions
        s.can_develop = True            #available for platform evaluation + developing
        s.game_name = {}                #temporal games names (Original was a string)
    
        s.consumer_prospects = []
        s.avg_prospects = []
        
    def announce_game(s,plat,market):
        s.can_develop = True
        #s.game_name = str(s.plats[0])+str(s.ID)+str(s.games)
        s.game_name[plat] = str(plat)+str(s.ID)+str(s.games)
        for pp in market.plats:
            if pp.ID == plat:
                pp.announced_games.append(s.game_name[plat])
        for c in market.cons:
            c.expect['game'][str(s.game_name[plat])] = 0
            c.experience['game'][str(s.game_name[plat])] = 0
        
        
    def develop_game(s,market,feat_type='uniform'):
    
        #get a 5 dimension array/vector
        if feat_type == 'uniform':
            game_factors = [1]*(s.feat_n)+[rd.uniform(-1,1) for i in range(10-s.feat_n)]
        elif feat_type == 'good':
            game_factors = [1  for i in range(10)]
        elif feat_type == 'bad':
            game_factors = [-1  for i in range(10)]
        else:
            game_factors = feat_type
        s.games += 1
        return game_factors
     
    def promote_game(s,plat,market):
        for c in market.cons:
            if rd.random() < 0.01:
                c.expect['game'][str(s.game_name[plat])] += 0.01
            
    def step(s,market):
        for g,g_units in s.unit_games.items():
            s.unit_games_ts[g].append(g_units)
            if len(s.unit_games_ts[g]) > 50:  #Wait for some market stability until saving weekly sales.
                s.unit_post10weeks[g] = g_units
                
        #New step (Old in Ver 2.0)
        if len(s.plats) > 0:
    #        print 'have plat '+str(s.plats)
            for p in s.plats:
                if s.dev_time[p] > 0:
                    s.can_develop = False
                    #if j <= 0 or len(market.plats) > len(s.plats): 
                        #s.can_develop = True
                    #    break
                    s.dev_time[p] -= 1
                    s.promote_game(p,market)
                    if s.dev_time[p] == 0:
                        for ps in market.plats:
                            if ps.ID == p:
                          #      print 'release!' 
                                #s.can_develop = True
                                s.old_adoption = s.old_adoption + [p]
                                s.plats.remove(p)
                                s.unit_games[str(s.game_name[p])] = 0
                                s.unit_games_ts[str(s.game_name[p])] = []
                                s.unit_games_time[str(s.game_name[p])] = market.steps
                                game_features = s.develop_game(market,feat_type=s.feat_type)
                                ps.games[s.game_name[p]] = game_features
                                market.games_features[s.game_name[p]] = game_features
                                market.games[s.game_name[p]] = [ps.ID,s.ID]
                            
                                ps.devs.remove(s.ID)
                                #ALSO GET HIM OUT OF PLATFORM
                            
                                break
        else:
   #         print 'want plat'
            s.can_develop = True
            if s.ID in market.licensed or market.steps > 10:
                if rd.random() < 0.1 and s.can_develop == True and len(market.plats) > 0 and len(s.plats) < s.license_lim: 
                    s.evaluate_plat(market)                                    #Random activation?
        
    def evaluate_plat(s,market):
        #Evaluate sample of consumers. For each platform check other devs, costs and expectation of consumers
        user_dev_ratio,dev_cost = [],[]
        
        for p in market.plats:
          
            if p.ID in market.retired_plat:
                pass
            else:
            #EVALUAN CON ADOPT
                user_adopt = 0
                #user_adopt = np.mean([c.adopt['plat'][str(p.ID)] for c in market.cons]) #if rd<s.bound
                user_expect = np.mean([float(c.expect['plat'][str(p.ID)]) for c in market.cons if rd.random() < s.bound])
                devs = len([this.devs for this in market.plats if p.ID == this.ID][0])
                user_dev_ratio.append(float(user_adopt+user_expect)/float(devs+1))
                #print devs
                #print '\n' 
                
                if devs < 0:
                    print user_adopt,user_expect
                    print '\n'
                #LAST CHANGE user_dev_ratio[-1] = 0
                                  
                #Missing license cost or dev cost
                if p.ID == '0_0':
                    dev_cost.append(0.02)
                else:
                    dev_cost.append(0)
        #print 'got some results'+str(len(user_dev_ratio))
        s.consumer_prospects.append(user_dev_ratio)

        #want = user_dev_ratio.index(max(user_dev_ratio))
        #Original model : how to get expected utility for developers (only for one platform!!, the MAX benefit)
        '''
        try:
            want_list = [float(user_dev_ratio[i]) - float(dev_cost[i]) for i in range(len(user_dev_ratio))]
            want_list2 = np.around(want_list,5)
            want_list2 = [float(i) for i in want_list2]
            want = want_list2.index(max(want_list2))
        except:
            return
        '''

        #Dev platform adoption with Multi - Homing (Chapter 3, Ver 3.0)
        try:
            want_list = [float(user_dev_ratio[i]) - float(dev_cost[i]) for i in range(len(user_dev_ratio))]
            want_list2 = list(np.around(want_list,5))
            want_list = [float(i) for i in want_list2]
            want_list2.sort(reverse=True)
            
            multi_homing = 2
            
            want = [want_list.index(ww) for ww in want_list2[:multi_homing]]   #Take the best two
                
        except:
            print 'Developer - Error on getting the best platforms.' 
        
        #want = want_list2.index(max(want_list2))
        
        exclusivity = iter(market.exclusivity)

        if len(want) > 0:
            for w in want:
                if market.plats[w].total_licenses > len(market.plats[w].devs):
                    if w not in market.retired_plat:
                        market.plats[w].devs.append(s.ID)
                        new_plat = market.plats[w].ID
                        s.plats.append(new_plat)
                        exc = exclusivity.next()
                        s.dev_time[new_plat] = rd.randint(40+exc,default_dev_time+exc)
                        s.announce_game(new_plat,market)
                        #print s.game_name[new_plat]
                        return
                else:
                    print 'Assign w didnt work' 


        #Original model : only take best platform
        '''                
        if user_dev_ratio.count(user_dev_ratio[want]) == 1:
            if market.plats[want].total_licenses > len(market.plats[want].devs):
                for p in market.plats:                        #Clean dev from all plat lists before re-assign
                    try:
                        if p.ID in s.plats:
                            p.devs.remove(s.ID)
                    except:
                         pass
                if want not in market.retired_plat:
                    market.plats[want].devs.append(s.ID)
                    s.plats = [market.plats[want].ID]
    
                    if market.plats[want].ID in s.plats:
                        s.dev_time = rd.randint(40,default_dev_time)
                    else:
                        s.plats.append(str(market.plats[want].ID))
                        s.dev_time = rd.randint(40,default_dev_time)
                    s.announce_game(market)
                    return
        else:
            print 'too many to decide'
        '''