# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:52:16 2016

@author: lulin
"""
import random as rd
import numpy as np

import numba
from numba import autojit
from numba import cuda
from numba import *

numba_mode = False

@numba.jit(nopython=True)                
def numba_interact2(npg,self_inf,self_expect):
  #  print 'numba'
    val = self_expect
    for n in npg:
        val = val + n*self_inf
    
    return float(val)/float(len(npg)+1)
    #val = float(sum([self_expect]+[n*self_inf for n in npg]))/float(len(npg)+1)

class theCons(object):
    def __init__(s,ID,netd,socialinfluence,Thresh_type='general',S=3,Q=10,T=0.6,preference='uniform',pref_n=2,perception=0):
        s.ID          = ID
        
        #Expectations
        s.expect,s.old_expect = {},{}              #s.expectative = [0 for j in range(S)]
        
        s.perception = rd.sample([0,1,2,3,4,5,6,7,8,9],perception)
        
        #Preferences
        if preference == 'uniform':
            s.preferences = [1]*pref_n+[rd.uniform(-1,1) for i in range(Q-pref_n)]  #5D (Q=5) vector preferences / Uniform distribution
        else:
            s.preferences = preference
         
        #Consumer habits and states
        s.consume_rate  = 2                    #Fixed product use (80-120)
        s.consumed = s.consume_rate               #s.consume memory (time left of consumption before needing another good)
        s.prospect = None
        s.playing = False
        
        #Purchase trigger
        if Thresh_type == 'general':      s.thresh  = T
        elif Thresh_type == 'individual': s.thresh  = rd.uniform(0,1)
        
        #Demography + Soc.Net.
        s.wealth = rd.random()                           #Uniform wealth (0-1). Placeholder
        s.influence = socialinfluence
        s.neighbors   = []
        s.position    = [0,0]
        
        #Given network
        s.netd = netd                                    #Agent social network degrees, circle of influence   
        
        #Purchases
        s.manuf,s.games = None,[]            #No target plat, no owned games / does not allow multihoming
        
        #Ver 3.0, plats are now a list for multi homing
        max_multihoming = 1
        s.plat = [None] * max_multihoming
                                                                  
        #Memory
        s.mem_plat = {}
        s.mem_dev = {}
        s.old_plat = []
        s.old_manuf= None
        s.consumeme = False
        s.store = {}           #Used for simultaneous update of expectations
        
        #Record keeping
        s.plat_steps = []
    #************************************************************************************************************
             
    #External functions (Called by Market()):
     
    #I - Set variables according to market (platforms and consumer population).
    def init_me(s,market):
        #Mean Expectation
        s.expect_m = 0
        
        #Expectation by platform / extendable for Expectation by Game (not used)
        s.expect = {'plat':{str(i):0 for i in range(len(market.plats))},'game':{}}
        s.old_expect = {'plat':{str(i):0 for i in range(len(market.plats))},'game':{}}
        
        #Adopt by platform / (Games(?))
        s.adopt = {'plat':{str(i):0 for i in range(len(market.plats))},'game':{}}
        
        #Experience by platform / (Games(?))
        s.experience = {'plat':{str(i):0 for i in range(len(market.plats))},'game':{}}
        
        #Purchase and temp expect
        s.bought = {'plat':{str(i):False for i in range(len(market.plats))},'game':{}}
        s.store  = {'plat':{str(i):0 for i in range(len(market.plats))},'game':{}}   #Temp Expect (for simultaneous update)
        
        #Network: Build list with surrounding neighbors - Erdos-Renyi (directed network)
        #s.neighbors = rd.sample([ids.ID for ids in market.cons],rd.randint(1,3))
        
        #CUDA
        #s.gpu_sub_interact2 = cuda.jit(device=True)(s.sub_interact2)
        
        #Chapter 3 - Part 2: NX Topology.
        #Circle
    #    if s.ID == 0:
    #        s.netd = [market.cons[-1].ID,s.ID+1]
    #    elif s.ID == market.cons[-1].ID:
    #        s.netd = [s.ID-1,0]
    #    else:
    #        s.netd = [s.ID-1,s.ID+1]
    #    s.position =[s.ID,0]
        #Thorus
        
        if type(s.netd) == list:
            #for node in s.netd:
            #    market.cons[node].neighbors.append(s.ID)
            #    market.netlinks += 1
            s.neighbors = s.netd[s.ID]
            
        elif s.netd == 'preferential':
            for node in market.cons:
                if node.ID != s.ID:
                    fThresh = float(1.0) / float((market.netlinks + 1) * (len(node.neighbors) + 1))
                    if(rd.random() <= fThresh):
                        node.neighbors.append(s.ID)
                        market.netlinks += 1
        
        elif s.netd == 'random':
            for node in market.cons:
                p = 0.01
                if rd.random() < p:
                    node.neighbors.append(s.ID)
                    market.netlinks += 1
                    




    #II - Update s.expect according to temporal expect (s.store), for simultaneous expect updates.
    def update(s,market):
        for p in market.announced_plat:
            if s.bought['plat'][p] == False:
                s.expect['plat'][p] = s.store['plat'][p]
            else:
                s.expect['plat'][p] = s.experience['plat'][p]
        #Update general expectative towards products (s.expect_m) - Currently working with MAX() (try Mean or Median)
        if len(market.announced_plat) > 0:
            s.expect_m = np.mean([float(s.expect['plat'][str(p)]) for p in market.announced_plat])
        
            #print 'Expect_m issue'
        for game in s.store['game'].keys():
            if game in s.games:
                s.expect['game'][game] = s.experience['game'][game]
            else:
                s.expect['game'][game] = s.store['game'][game]
    

    #III - Main step function for every time-tick. Activation schedule depends on Market()).
    #@numba.jit(nopython=True) #@autojit
    def step(s,market):
        #0. Interact with others
        #Activation of interaction. Be influenced by others (adopter and non-adopters)
        #if rd.random() < 1:
        s.interact(market)
        
        #if rd.random() < 0.2:
        #    print s.plat
        
        #1. Remember status
        s.plat_steps.append(s.plat)
        
        #2. Adopter behavior: search and consume games (Adopter).
        #Main step changed for Multi Homing extension on Chapter 3. Check Ver 2.0 for older consumer behaviors
        ix = -1
        for splat in s.plat:
            ix += 1                   #Index
            if splat != None:                           
                s.consumed -= 1    
                
                #Not playing?
                if s.playing == False:
                    #Want to play more games?
                    
                    #Get games in my platform
                    for ps in market.plats:
                        if ps.ID == splat:
                            games_in_plat = ps.games.keys()
                                    
                    if np.mean([s.expect['game'][g] for g in games_in_plat if g not in s.games]) < 0:
                        s.adopt['plat'][splat] = 0
                        s.old_plat.append(splat)   
                        s.old_manuf = str(splat)[0]
                        for platform in market.plats:
                            if platform.ID == splat:
                                platform.users -= 1
                                break
                        s.plat[ix],s.consumeme,s.playing = None,False,False
                    
                    elif rd.random() < s.wealth:
                        #print 'no calls to devs yet'
                        s.choose_game_to_buy(market)
                        #print 'i just called'
                    
                #50% chance of stop playing.
                if s.playing and rd.random() < 0.5:
                    s.playing = False
                    
                #If already consumed then leave platform adoption
                if s.consumed <= 0:                      
                    s.adopt['plat'][splat] = 0
                    s.old_plat.append(splat)
                    s.old_manuf = str(splat)[0]
                    for platform in market.plats:
                        if platform.ID == splat:
                            platform.users -= 1
                            break
                    s.prospect = None
                    s.plat[ix],s.consumeme = None,False
                    s.playing = False
    
            #3. Evaluate otherwise (Non Adopter).
            elif splat == None:                                        
                #Off-purchase experience - Sharing products, store demos, etc.
                #if rd.random() < 0.1 and len(market.released_plat) > 0:             
                #    #Generate temporal plat and capture experience.
                #    s.plat = market.plats[rd.randint(0,len(market.released_plat)-1)].ID
                #    s.experience_plat(market)
                #    s.plat = None
                
                #If median expectation is high enough, search platforms (prospect) to adopt.
    
                if rd.random() < s.expect_m and len(market.released_plat) > 0:
                    s.compare_plat(market)
                
                #Have a prospect? Try to buy it!
                if s.prospect != None:
                    s.go_buy(ix,market)
                    
            elif type(splat) not in [None,str]:
                print type(splat)

    #***************************************************************************************************************

    #Internal functions (called by self to process interaction, evaluation and consumption.)
    
    #I - Interact with others and change expectation (s.expect)
    #@autojit
    def interact(s,market):
        #Go through non-purchased plats and update expectations for plat and its games.
        marketpos = market.pos_ID
        for p in market.announced_plat:
            
            s.sub_interact1(market,marketpos,p)


        non_played_games = [k for k,v in s.expect['game'].items() if k[:3] not in s.old_plat and s.experience['game'][k] == 0][:10]
        if len(non_played_games) > 0 and numba_mode == False:
            s.store = s.sub_interact2(market.cons,marketpos,non_played_games)
        elif numba_mode == True:
            game_val = []
            for g in non_played_games:
                game_val.append([float(market.cons[n].expect['game'][g]) for n in s.neighbors])
            for g in range(len(non_played_games)):
                s.store['game'][non_played_games[g]] = numba_interact2(game_val[g],s.influence,s.expect['game'][non_played_games[g]])
               # print 'not numba'
    
    #@autojit 
    def sub_interact1(s,market,marketpos,p):
            if s.bought['plat'][p] == False:
            
                #OLD WAY
                #     if s.experience['plat'][str(p)] == 0 and p not in market.retired_plat and p not in s.old_plat: # and p not in market.retired_plat:
                #         s.expect['plat'][str(p)] = np.clip(np.average(s.expect['plat'][str(p)]+np.average([market.cons[n].expect['plat'][str(p)] for n in s.neighbors])*.1),-1,1) #s.influence),-1,1)
                #Natural decay / post-lifetime            
                #    if s.experience['plat'][str(p)] != 0 and p in s.old_plat: #or p in market.retired_plat :
                #        s.expect['plat'][str(p)] = np.clip(s.expect['plat'][str(p)]*.95,-1,1)
                
                #NEW WAY
                #my_neighbors_position = [int(market.pos_ID[market.pos_ID.index == id_][0]) for id_ in s.neighbors]  #This calls ID to pos_ID and returns position
                #others_expectatives = [market.cons[a].expect['plat'][p]*s.influence for a in my_neighbors_position]                                             #Get expect from neighbors
                others_expectatives = [market.cons[int(marketpos[id_])].expect['plat'][p]*s.influence for id_ in s.neighbors]                                             #Get expect from neighbors
                
                tot = 0
                for e in others_expectatives:
                    tot = tot + e
                tot = tot + s.expect['plat'][p]
                s.store['plat'][p] = float(tot) / float(len(others_expectatives)+1)
                #s.store['plat'][p] = float(sum(others_expectatives+[float(s.expect['plat'][p])]))/float(len(others_expectatives)+1)
                #s.store['plat'][p] = np.clip(s.store['plat'][p],-1,1)
            
                if market.steps in range(45,65):
                    market.debug_expect[market.steps][s.ID] = [s.expect['plat'][p],others_expectatives,s.store['plat'][p]]
    
  #  @numba.jit(nopython=True)
    def sub_interact2(s,marketcons,marketpos,non_played_games):
            #non_played_games = [k for k,v in sorted(s.expect['game'].iteritems(), key=lambda (k,v): (v,k), reverse=True) if k[:3] not in s.old_plat and s.experience['game'][k] == 0][:10]
            
            for gamename in non_played_games:
                # main s.store['game'][str(gamename)] = float(sum([s.expect['game'][str(gamename)]]+[marketcons[n].expect['game'][gamename]*s.influence for n in s.neighbors]))/float(len(s.neighbors)+1)
                #   s.store['game'][str(gamename)] = np.mean([s.expect['game'][str(gamename)]]+[marketcons[n].expect['game'][gamename]*s.influence for n in s.neighbors])
                tot = 0
                for n in s.neighbors:
                    tot = tot + marketcons[n].expect['game'][gamename]*s.influence
                tot = tot + s.expect['game'][gamename]
                s.store['game'][str(gamename)] = float(tot)/float(len(s.neighbors)+1)
                #    self_store[str(gamename)] = float(sum([self_expect[str(gamename)]]+[1*self_influence for n in self_nn]))/float(len(self_nn)+1)
                #temp_neighbors = {gamename:[market.cons[n].expect['game'][gamename]*s.influence for n in s.neighbors] for gamename in non_played_games}
            
            return s.store    
                #s.store['game'][str(game)] = np.mean([s.expect['game'][str(game)]]+temp_neighbors[str(game)])
                    #if s.experience['game'][str(game)] != 0:
                    #    s.store['game'][str(game)] = float(s.experience['game'][str(game)])/1.5
    

        
                
    def post_evaluation(s,product_features):        
        normalized_feat = product_features / np.linalg.norm(product_features)
        normalized_pref = s.preferences / np.linalg.norm(s.preferences)
        return np.dot(normalized_feat,normalized_pref)


    def perceive(s,game_value):
        if len(s.perception) > 1:
            perceived_features = [game_value[i] for i in range(len(game_value)) if i in s.perception]
            relevant_preferences = [s.preferences[i] for i in range(len(s.preferences)) if i in s.perception]
            normalized_feat = perceived_features / np.linalg.norm(perceived_features)
            normalized_pref = relevant_preferences / np.linalg.norm(relevant_preferences)
            utility = np.dot(normalized_feat,normalized_pref)
            return utility
        else:
            return 'boop'
        
    #II - Experience a platform (via demo or purchase).       
    def experience_plat(s,market,ix,target_plat):

        #Get product features
        plat_features = market.plat_features[target_plat]
        
        #OLD EXPERIENCE
        #exp = np.sum([float(s.preferences[i]*plat_values[i]) for i in range(len(s.preferences))])
        
        #Normalized Dot Product for Experience
        exp = s.post_evaluation(plat_features)
        
        s.experience['plat'][str(s.plat[ix])] = exp
        s.expect['plat'][str(s.plat[ix])] = exp 
        s.store['plat'][str(s.plat[ix])] = exp 
        
        return exp
    
    
    #III - Expectation and other parameters such as Technology are used to compare/evaluate.
    def compare_plat(s,market):
        #print ' compare begin' 
        if len(market.plats) > 0:                         #If there are any platforms, try and check them
            b,bids,td=[],[],len(market.devs)              #b: benefit, bids: benef. ids, td: total devs
            
            #Tech = step^2 / Platforms release day^2 is platform tech. Same tech for all platforms
            tech = [float(i.release_d**2) for i in market.plats if i.ID not in market.retired_plat and i.ID in market.released_plat]
            #Get normalized technology for released platforms
            norm_tech = {str(i.ID):float(i.release_d**2)/max(tech) for i in market.plats if i.ID not in market.retired_plat and i.ID in market.released_plat}
            
            #Get expected/evaluated benefit
            for target in market.plats:
                
                #Cant evaluate those that already bought
                if s.bought['plat'][target.ID] == False:
                    unplayed_games = len([i for i in target.announced_games if i not in s.games])
                    total_games = len(target.announced_games)
                    
                    if target.ID in market.released_plat and target.ID not in market.retired_plat:     
                        ben = ((float(unplayed_games)/(total_games+1))+float(s.expect['plat'][str(target.ID)]))+norm_tech[target.ID]
                                   #/s.expect_m #*norm_tech[target.ID])
                    #    print ben
                        tben = ben-target.consumerprice
                     #   print tben
                     #   print '\n' 
                        
                        if tben > 0:
                            #print tben,'\n'
                            b.append(tben)
                            bids.append(target.ID)

            #Is there any good candidate?
            if len(b) > 0:
                #From all benefits use unique max benefit, adopt that platform and reject others
                want = bids[b.index(max(b))]
                
                if b.count(b[b.index(max(b))]) == 1:
                    s.prospect = str(want)
      #  print ' compare end' 

    #IV - Purchase platform.                    
    def go_buy(s,ix,market):
        if rd.random() < s.wealth:
            s.plat[ix] = str(s.prospect)
            
            #Initial usage motivation
            s.consumed = 8
            s.consumeme = True
            
            for platform in market.plats:
                if platform.ID == s.plat[ix] and platform.ID not in market.retired_plat and platform.ID not in s.old_plat:
                    s.bought['plat'][s.plat[ix]] = True
                    s.adopt['plat'][s.plat[ix]] = 1
                           
                    platform.hwsales_step += 1
                    platform.users += 1
                    
                    exp = s.experience_plat(market,ix,s.plat[ix])

                    market.disadoptersall.append(exp)
                    
                    if exp < 0:
                        s.adopt['plat'][s.plat[ix]] = 0
                        s.old_plat.append(s.plat[ix])
                        s.old_manuf = str(s.plat[ix])[0]
                        platform.users -= 1
                        s.plat[ix],s.consumeme = None,False
                        s.playing = False
                    break
            s.prospect = None
    
    #V - Purchase games.        
    def choose_game_to_buy(s,market):
        try:
            for ps in market.plats:
                    if ps.ID in s.plat:
                        games_in_plat = ps.games.keys()
                        
            #game_val = {g:s.expect['game'][g] for g in games_in_plat if g not in s.games}
            game_val = [k for k,v in sorted(s.expect['game'].iteritems(), key=lambda (k,v): (v,k), reverse=True) if k not in s.games and v > 0 and k[:3] in s.plat]
            if len(game_val) == 0:
                buygames = [None]
            else:
                buygames = game_val
            '''
    #Chapter 3 - Part 2: Consumer Evaluation            
            best_game = [None,0]
            #Test for 3,5, ten is a lot of adoption on platforms and cool fits
            for buygame in buygames[:10]:
                if buygame != None and buygame in market.games.keys() and s.plat == market.games[buygame][0]:   
                        #Get game features
                        game_value = market.games_features[buygame]
                        
                        s.perception = [3,4,5]
                        perceived_features = [game_value[i] for i in range(len(game_value)) if i in s.perception]
                        relevant_preferences = [s.preferences[i] for i in range(len(s.preferences)) if i in s.perception]
                        normalized_feat = perceived_features / np.linalg.norm(perceived_features)
                        normalized_pref = relevant_preferences / np.linalg.norm(relevant_preferences)
                        utility = np.dot(normalized_feat,normalized_pref)
                        if utility > best_game[1]:
                            best_game[0] = buygame
                            best_game[1] = utility
            
            if best_game[1] > 0:
            '''
            buygame = buygames[0]
                #Old part
            if buygame != None and buygame in market.games.keys() and market.games[buygame][0] in s.plat:   
                #Get game features
                game_value = market.games_features[buygame]
                
                utility = s.perceive(game_value)
                if utility > 0:#Buy game (modify my library and developer sales)
    #REMEMBER            buygame = best_game[0]
                    s.games.append(buygame)
                    dev = market.games[buygame][1]
                    market.devs[dev].unit_games[buygame] += 1
                    
                    #Get game features 
                    #game_value = market.games_features[buygame]
                    
                    #Experience the game (get dot product utility)    
                    exp = s.post_evaluation(game_value)    #    s.experience['game'][buygame] = np.sum([float(s.preferences[i])-float(game_value[i]) for i in range(len(s.preferences))])     
                    s.experience['game'][buygame] = exp
                    s.expect['game'][buygame] = exp
                                
                    #If experience is positive, consume more and start playing boi
                    if s.experience['game'][buygame] > 0:
                        s.consumed += s.consume_rate
                        s.playing = True
                elif utility < 0:
                    s.experience['game'][buygame] = utility
                    s.expect['game'][buygame] = utility
                    
                    
                #break
                
            #Select game with higher expectation
            #for k,v in game_val.items():
            #    if v == max(game_val.values()) and v > 0 and k not in s.games:     #If best expectation is positive
            #        buygame = k
            #        break
            '''
            #If game still in the market
            max_ = 3
            for buygame in buygames:
                print buygame
                if max_ == 0:
                    break
                else:
                    max_ -= 1
                if buygame != None and buygame in market.games.keys() and s.plat == market.games[buygame][0]:
                    
                    #Buy game (modify my library and developer sales)
                    s.games.append(buygame)
                    dev = market.games[buygame][1]
                    for d in market.devs:
                        if d.ID == dev:
                            d.unit_games[buygame] += 1
                            break
                    
                    #Get game features
                    game_value = market.games_features[buygame]
                    
                    #Experience the game (get dot product utility)    
                    exp = s.post_evaluation(game_value)    #    s.experience['game'][buygame] = np.sum([float(s.preferences[i])-float(game_value[i]) for i in range(len(s.preferences))])     
                    s.experience['game'][buygame] = exp
                    s.expect['game'][buygame] = exp
                                
                    #If experience is positive, consume more and start playing boi
                    if s.experience['game'][buygame] > 0:
                        s.consumed += s.consume_rate
                        s.playing = True
                '''
                    

        except Exception as e:
            pass #print e