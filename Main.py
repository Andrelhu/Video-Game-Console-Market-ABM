#!/usr/bin/python2.4
# -*- coding: utf-8 -*-
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
import winsound
mpl.rc('savefig',dpi=100)
from matplotlib import animation

import time
import theMarket
from theCases import *
import seaborn as sns
import pandas as pd

#Network related
import networkx as nx
from collections import Counter

#CUDA
from numba import autojit


#To run:
def simple_run():
    market = G()
    #market = market[0].values()[0]
    #return market

Basic_Test = {'name': 'Test',
         'plats': 4,
         'devs' : 250,
         'cons' : 5000,
         'step' : 300,
         'gens' : 1,
         #'dates': [55,56,57,58], #sin Pong(75)
         #'dates': [55,79,103],
         'dates': [55,91,127,163], #sin Pong(75)
         'all'  : True,
         'soc'  : 0.3,
         'media': 0.7,
         'bound': 1,
         'pref_n': 2,
         'feat_n': 2}

#Ver 3.0: 
# - Extensions:
    #1 Adjustable preferences
    #2 Adjustable features
    
#Profiler, taken from https://osf.io/upav8/
import cProfile, pstats #old:, io
#Modified for 2.7
from io import BytesIO as StringIO

def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = StringIO() #old: io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())[:1500]
        return retval

    return inner
    
#Main functions

#I - Quick Main() with Network plot.

def G(case=Basic_Test,info=False,experiments=1,exp_plot=True,plot_all=False,preferences='uniform'):
    if info:
        print '\nRunning case: "'+str(case['name'])+'".\n'
        print 'Consumers: '+str(case['cons'])
        print 'Publishers: '+str(case['devs'])
        print 'Platforms: '+str(case['plats'])
        print 'Peer factor at: '+str(case['soc'])
        print 'Media factor at: '+str(case['media'])
        print 'Repeat '+str(experiments)+' times for experiments.'
          
    sim,temp = __main__(case,experiments,exp_plot,preferences=preferences,features='uniform')
    #get_network(plot=True,snaps=11)
    return sim,temp

#II - Main(), sets parameters and runs run_sim()
#@profile
def __main__(case,experiments,exp_plot,plot_all=True,preferences='uniform',features='uniform'):
    plats = case['plats']
    devs  = case['devs']
    cons  = case['cons']
    step  = case['step']
    exper = case['name']
    socinf = case['soc']
    massinf = case['media']
    devinf = case['bound']
    pref_n = case['pref_n']
    feat_n = case['feat_n']
    t0 = time.time()
    
    start_at = 0
    preftag = None
    
    target_par = ['cum_avg','cum_med','plat_rank','dev_rank','releases','dev_plats','expect']
    experiment_d = {i:[] for i in target_par}
    for i in range(experiments):
            try:
                experiment = []
                print '\n Running experiment # '+str(i+1)
                sim,plt_res = run_sim(plats,devs,cons,
                        netd='random',
                        dev_bnd=devinf,
                        generations=1,
                        release_dates=case['dates'],
                        socinf=socinf,
                        price=[1]*plats,
                        quality=[1]*plats,
                        steps=step,
                        plot=True,
                        name=exper,
                        massinf=massinf,
                        plot_allb=plot_all,
                        preferences=preferences,
                        features=features,
                        pref_n=pref_n,
                        feat_n=feat_n)
                        #preftag=preftag)
                
                rank = []
                n_targeted = range(125)
                targeted = []
                temporal,temp2 = [],[]
                for j in sim.devs:
                    rank.append(sum(j.unit_games.values()))
                    if int(j.ID) in n_targeted:
                        targeted.append(sum(j.unit_games.values()))
                rank.sort(reverse=True)
                targeted = [targeted[iii]+float(iii)/1000 for iii in range(len(targeted))]
                print len(targeted)
                t_taken = []
                for k in range(len(rank)):
                    token = False
                    for target in targeted:
                        if target >= rank[k] and target not in t_taken:
                            #print k+1
                            t_taken.append(target)
                            temporal.append(k+1)
                            token = True
                            break
                    if token == False:
                        temp2.append(k+1)
                print len(t_taken)
                sim = experiment_info(sim,step,temporal,temp2)    
                
                #experiment.append({sim:sim.experiment})
                experiment.append({i:sim.experiment})
                dfexp = pd.Series(experiment)
                dfexp.to_pickle('Simulations/Exp'+str(exper)+'_'+str(start_at+i+1))
                for pl in plt_res.keys():
                    experiment_d[pl].append(plt_res[pl])
            except Exception as e:
                print 'Error in sim #'+str(i+1)+' : '+str(e)
    
    if exp_plot:
        plot_experiment(experiment_d,devs)
        
    #export_nx_csv(sim)
    #central_weekly(sim,report=True)
    
   # print 'Simulaton complete. \nDuration: '+str(time.time()-t0)+' segs.'
    print ''
    get_alert()
    print temporal
    return dfexp,temporal #former "sim"

@autojit
def create_pref(population):
    cons = range(0,population)
    netlinks = 0
    agents = []
    
    for me in cons:
            neighborhood = []
            for node in cons:
                fThresh = float(1.0) / float((netlinks + 1) * (len(neighborhood) + 1))
                if(rd.random() <= fThresh):
                    if me != node:
                        neighborhood.append(node)
                        netlinks += 1
            #neighborhood.append(rd.randint(0,population))
            agents.append(neighborhood)
    return agents

@autojit
def create_rand(population):
    cons = range(0,population)
    netlinks = 0
    agents = []

    for me in cons:
            p = 0.005
            neighborhood = []
            for node in cons:
                if rd.random() < p:
                    neighborhood.append(node)
                    netlinks += 1
            agents.append(neighborhood)
    return agents

#III - Run_sim() actual simulation of N steps according to parameters.
def run_sim(p,d,c,price,quality,steps=300,netd='random',mkttime=100,plot=True,dev_bnd=1,generations=1,release_dates=None,socinf=0.1,massinf=1,name='Normal run',plot_allb=True,preferences='uniform',features='uniform',pref_n=2,feat_n=2):
    print ''
    print 'Sim begins: '+str(name)+'\nManagers : '+str(p)+'\nDevelopers : '+str(d)+'\nConsumers : '+str(c)
    print 'Peer factor: '+str(socinf)+'. Media factor: '+str(massinf)
    global market
    
    t0 = time.time()
    
    #netdd = create_rand(c)
    netdd = create_pref(c)
    
    market = theMarket.Market(p,d,c,price,netdd,mkttime,dev_bnd,generations,quality,steps,release_dates,socinf=socinf,massinf=massinf,preferences=preferences,features=features,pref_n=pref_n,feat_n=feat_n)
    market.init_agents(market)
    market.pos_to_ID(market)
    
    t1 = time.time()
    print 'Init done in T: '+str(t1-t0)+' secs.'
    for s in range(steps):
        market.step(market)
    #End of Simulation procedure
    t2 = time.time()
    
    print 'Sim in T: '+str(t2-t1)+' secs.'
    print  ''
    print 'Total '+str(t2-t0)+' secs.'
    
    if plot_allb == True:
        results = plot_all(market,steps,d)
        #get_network(plot=True)
    else:
        results = False
    
    return market,results

#Functions for simulation data, visualization, and analysis:
    
#Get empirical data.
def get_empirics(plot=False):
    par1df = pd.read_pickle('Empirics/Empirics_Parameter1') #platform distribution
    par2df = pd.read_pickle('Empirics/Empirics_Parameter2') #cumulative adoption
    par3df = pd.read_pickle('Empirics/Empirics_Parameter3') #
    par4df = pd.read_pickle('Empirics/Empirics_Parameter4')
    
    if plot:
        total_platforms = list(par1df.Distro_gen)[-1]
        plt.title('Average sale distribution (platform).')
        plt.bar(np.arange(0,len(total_platforms)),total_platforms,width=1)
        plt.show()
        
        plt.title('Average adoption (cumulative).')
        for k in ['AVG','STD','MED']:
            plt.plot(par2df[k],label=k)
        plt.legend()
        plt.show()
    
    return par1df,par2df,par3df,par4df


#Get main data from each simulation run.
def experiment_info(market,steps,temporal,temp2):
    for pl in market.announced_plat:
        #Get Consumer adoption.
        temp_record = [0]*(steps-len(market.record[str(pl)]))+market.record[str(pl)]
        market.experiment['adopters'][pl] = temp_record

        #Get Developers adoption.
        temp_dev_rec = [0]*(steps-len(market.dev_rec[str(pl)]))+market.dev_rec[str(pl)]
        market.experiment['developers'][pl] = temp_dev_rec

    for pl in market.plats:
        #Get Cumulative adoption (consumer).
        max_hw = max(pl.hwsales)+0.01
        csales = [float(plp)/max_hw for plp in pl.hwsales if float(plp)/max_hw != 1 and float(plp)/max_hw != 0] 
        ccsales = [sum(csales[(i-1)*(len(csales)/20):i*(len(csales)/20)]) for i in range(1,21)]
        max_ccsales = max(ccsales)+0.01
        market.experiment['cumulative'][pl] = [0]+[float(cc)/max_ccsales for cc in ccsales]
                         
    #Get Totals via Weekly
    market = central_weekly(market,experiment=True)
    
    a = [sum(i.unit_post10weeks.values()) for i in market.devs]
    a.sort(reverse=True)
    market.experiment['dev_rank'] = a
    
    rel_games = []
    for pl in market.plats:
        rel_games.append(len(pl.games))
    rel_games.sort(reverse=True)
    market.experiment['releases'] = rel_games
                 
    mean,stdev = get_weekly_games()
    market.experiment['weekly_mean'] = mean
    market.experiment['weekly_std'] = stdev
    
    #ver3.0 rankings
    market.experiment['targetrank'] = [temporal,temp2]
    #ver3.0 NETWORK: GET A NETWORK VISUALIZATION
    
    #this sample gives too many clickes
    #sample = rd.sample(market.cons,int(len(market.cons)/10))
    
    #try this for big clusters
    sample = rd.sample(market.cons,int(len(market.cons)/10))
    node , target , product = [] , [], {}
    for c in sample: #sampled consumer
        if len(c.old_plat) >=1:
            product[c.ID] = c.old_plat[-1]
        else:
            product[c.ID] = 'None'
        for n in c.neighbors:
            node.append(c.ID)     #get his links
            target.append(n)
            for e in market.cons:      #for all his links get their neighbbors
                if e.ID == n:
                    if len(e.old_plat) >=1:
                        product[e.ID] = e.old_plat[-1]
                    else:
                        product[e.ID] = 'None'
                    for nn in e.neighbors:
                        node.append(n)
                        target.append(nn)
                        for nnn in market.cons:
                            if nnn.ID == nn:
                                if len(nnn.old_plat) >=1:
                                    product[nnn.ID] = nnn.old_plat[-1]
                                else:
                                    product[nnn.ID] = 'None' 
    #df = pd.DataFrame(node)
    #df['Target'] = target
    #df.columns = ['Source','Target']
    print 'done'
    market.experiment['network'] = [node,target,product]
    return market

#Visualization functions
               
def plot_experiment(experiment,d):
        print '\n\n\n@@@@@@@@@@@@@@@@@@@@@@\n\nEXPERIMENTS RESULTS\n\n'
        par1,par2,par3,par4 = get_empirics()
        
    #Plot simulation (Adopters and developers by platform (colors)):
        sns.set(rc={'axes.facecolor':'#E6E6FA', 'figure.facecolor':'white'})
        names = {'0':'Alpha ','1':'Beta ','2':'Gamma ','3':'Theta','4':'Delta','M':'Average Exp.','D':'Std. Dev. of Exp.'}
        gens = {'0':'Gen I','1':'Gen II','2':'Gen III','M':'','D':''}
        line = {'0_0':'--','0_1':'--','1_0':'--','1_1':'--','2_0':'--','2_1':'--','3_0':'--','3_1':'--','4_0':'--','4_1':'--','M':'--','M_SD':'-'}
        
    #Plot cumulative adopters by platform at given step       
        plt.figure(figsize=(6,4))
        plt.title('Cumulative Purchase by Platform (Average)')
        cdict = get_col()        
        for ppp in experiment['cum_avg']:
            plt.plot(ppp,alpha=0.25,linewidth=0.8)    
        plt.plot(np.average(experiment['cum_avg'],axis=0),label='Sim.Avg.',alpha=0.5,linewidth=1.8,)
        plt.plot([float(i)/max(par2['AVG']) for i in par2['AVG']],label='Emp.Avg.',alpha=0.5,linewidth=1.8,color='r')
        plt.ylabel('Units')
        plt.xlabel('Steps')
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
                
        plt.figure(figsize=(6,4))
        plt.title('Cumulative Purchase by Platform (Median)')
        cdict = get_col()
        for ppp in experiment['cum_med']:
            plt.plot(ppp,alpha=0.25,linewidth=0.8)   
        plt.plot(np.average(experiment['cum_med'],axis=0),label='Sim.Med.',alpha=0.5,linewidth=1.8)
        plt.plot([float(i)/max(par2['MED']) for i in par2['MED']],label='Emp.Med.',alpha=0.5,linewidth=1.8,color='r')
        plt.ylabel('Units')
        plt.xlabel('Steps')
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
    #Plot total games per platform
        plt.figure(figsize=(6,4))
        plt.title('Platform rank')
        par11 = [float(i)/max(par1) for i in par1]
        
        exp11 = np.average(experiment['plat_rank'],axis=0)
        print experiment['plat_rank'], exp11
        
        plt.bar(np.arange(0,len(exp11)),exp11,label='Sim Rank',alpha=0.8,width=0.5,color='b')
        plt.bar(np.arange(0.5,len(par11)+0.5),par11,label='Emp Rank',alpha=0.8,width=0.5,color='r')
        plt.legend()
        plt.show()
    
    #Distribution of games
        plt.figure(figsize=(6,4))
        plt.title('Developer total unit sales.')
        ppar3 = [float(ii)/max(par3) for ii in par3]
        aa = np.average(experiment['dev_rank'],axis=0)
        plt.plot(np.arange(0,len(aa)),aa,label='Sim.',alpha=0.8,linewidth=1.8)
        plt.plot(np.arange(0,len(ppar3[:len(aa)])),ppar3[:len(aa)],label='Emp.',alpha=0.8,linewidth=1.8,color='r')
        plt.xlabel('Rank order of developer sales')
        plt.ylabel('Units')
        plt.ylim(0,max(aa))
        plt.show()
        
    #Developers by platform
        plt.figure(figsize=(6,4))
        plt.title('Developers by Platform')
        cdict = get_col()
        dev_plats = experiment['dev_plats']
        for l in range(len(dev_plats[0])):
            plt.plot(np.average([devs[l] for devs in dev_plats],axis=0),linewidth=1.5,alpha=0.6)
        plt.ylim(0,d)
        plt.ylabel('Developers')
        plt.xlabel('Steps')
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
                
    #Total games per platform
    #    plt.figure(figsize=(6,4))
     #   plt.title('Releases by Platform')
     #   cdict = get_col()
    #    rel_games = experiment['releases'] 
    #    par44 = [float(i)/max(par4) for i in par4]
    #    plt.bar(np.arange(0,len(rel_games)),np.average(rel_games,axis=0),alpha=0.6,width=1,label='Sim.')
    #    plt.plot(par44,alpha=0.8,label='Emp.')
    #    plt.ylabel('Releases')
    #    plt.xlabel('Rank')
    #    plt.xticks()
    #    plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
    #    plt.show()

    #Expectations by platform
        plt.figure(figsize=(6,4))
        plt.title('Platform Expectation by Console and Average(M).')
        cdict = get_col()
        plt.plot(np.average(experiment['expect'],axis=0),linewidth=1.5,alpha=0.6)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()    
              
#I - Plot all relevant parameter time series.
def plot_all(market,steps,d):
        par1,par2,par3,par4 = get_empirics()
        test = {}
        
    #Plot simulation (Adopters and developers by platform (colors)):
        sns.set(rc={'axes.facecolor':'#E6E6FA', 'figure.facecolor':'white'})
        names = {'0':'Alpha ','1':'Beta ','2':'Gamma ','3':'Theta','4':'Delta','M':'Average Exp.','D':'Std. Dev. of Exp.'}
        gens = {'0':'Gen I','1':'Gen II','2':'Gen III','M':'','D':''}
        line = {'0_0':'--','0_1':'--','1_0':'--','1_1':'--','2_0':'--','2_1':'--','3_0':'--','3_1':'--','4_0':'--','4_1':'--','M':'--','M_SD':'-'}

    #Expectations by platform
        plt.figure(figsize=(6,4))
        plt.title('Mean Expectation.')
        cdict = get_col()
        for k in market.expec_rec.keys():
            market.experiment['expectatives'][k] = market.expec_rec[k]
            if len(market.expec_rec[k]) > 5 and k in ['M','M_SD']:
                labelthis = str(names[str(k)[-1]])+str(gens[str(k)[-1]])
                plt.plot(market.expec_rec[k],color=cdict[str(k)[0]],label=labelthis,alpha=0.6,linestyle=line[k])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    
        test['expect'] = market.expec_rec['M']

    #Plot Adopters by Platform at given step
        plt.figure(figsize=(6,4))
        plt.title('Adopters by Platform')
        cdict = get_col()
       
        plat_totals= []
        for pl in market.announced_plat:
            temp_record = [0]*(steps-len(market.record[str(pl)]))+market.record[str(pl)]
            market.experiment['adopters'][pl] = temp_record
            labelthis = str(names[str(pl)[0]])+str(gens[str(pl)[-1]])
            plt.plot(temp_record,color=cdict[str(pl)[0]],label=labelthis,linewidth=1.5,alpha=0.6,linestyle=line[pl])
            plat_totals.append(sum(temp_record))
        plt.ylim(0,len(market.cons))
        plt.ylabel('Adopters')
        plt.xlabel('Steps')
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


    #Plot total platforms
        plt.figure(figsize=(6,4))
        plt.title('Platform rank')
        plat_totals.sort(reverse=True)
        plat_totals = plat_totals+[0]*(len(par1)-len(plat_totals))
        plat_totals = [float(i)/max(plat_totals) for i in plat_totals]
        par11 = [float(i)/max(par1) for i in par1]
        plt.plot(plat_totals[:len(par1)],label='Sim Rank',alpha=0.8)
        plt.plot(par11,label='Emp Rank',alpha=0.8)
        plt.legend()
        plt.show()
        
        test['plat_rank'] = plat_totals[:len(par1)]
        
    #Plot total games per platform
        plt.figure(figsize=(6,4))
        plt.title('Platform rank')
        plat_totals.sort(reverse=True)
        plat_totals = plat_totals+[0]*(len(par1)-len(plat_totals))
        plat_totals = [float(i)/max(plat_totals) for i in plat_totals]
        par11 = [float(i)/max(par1) for i in par1]
        plt.plot(plat_totals[:len(par1)],label='Sim Rank',alpha=0.8)
        plt.plot(par11,label='Emp Rank',alpha=0.8)
        plt.legend()
        plt.show()
        
    #Plot cumulative adopters by platform at given step
        plt.figure(figsize=(6,4))
        plt.title('Cumulative Purchase by Platform')
        cdict = get_col()
        
        average_adopters = []
        for pl in market.plats:
            range_steps = 15
            labelthis = str(names[str(pl.ID)[0]])+str(gens[str(pl.ID)[-1]])
            max_hw = max(pl.hwsales)+0.01
            csales = [float(plp)/max_hw for plp in pl.hwsales if float(plp)/max_hw != 1 and float(plp)/max_hw != 0] 
            ccsales = [sum(csales[(i-1)*(len(csales)/range_steps):i*(len(csales)/range_steps)]) for i in range(1,range_steps+1)]
            max_ccsales = max(ccsales)+0.01
            market.experiment['cumulative'][pl] = [0]+[float(cc)/max_ccsales for cc in ccsales]
            temp_adopters = [0]+[float(cc)/max_ccsales for cc in ccsales]
            average_adopters.append(temp_adopters)
            plt.plot(temp_adopters,color=cdict[str(pl.ID)[0]],label=labelthis,linewidth=1,alpha=0.3,linestyle=line[pl.ID])
        plt.ylabel('Units')
        plt.xlabel('Steps')
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
        plt.figure(figsize=(6,4))
        plt.title('Cumulative Purchase by Platform')
        cdict = get_col()        
        plt.plot(np.average(average_adopters,axis=0),label='Sim.Avg.',alpha=0.5,linewidth=1.8)
        plt.plot([float(i)/max(par2['AVG']) for i in par2['AVG']],label='Emp.Avg.',alpha=0.5,linewidth=1.8)
        plt.ylabel('Units')
        plt.xlabel('Steps')
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
        test['cum_avg'] = np.mean(average_adopters,axis=0)
        market.experiment['cum_avg'] = np.mean(average_adopters,axis=0)
        
        plt.figure(figsize=(6,4))
        plt.title('Cumulative Purchase by Platform')
        cdict = get_col()
        plt.plot(np.median(average_adopters,axis=0),label='Sim.Med.',alpha=0.5,linewidth=1)
        plt.plot([float(i)/max(par2['MED']) for i in par2['MED']],label='Emp.Med.',alpha=0.5,linewidth=1)
        plt.ylabel('Units')
        plt.xlabel('Steps')
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
        test['cum_med'] = np.median(average_adopters,axis=0)
        market.experiment['cum_med'] = np.median(average_adopters,axis=0)
        
    #Developers by platform
        plt.figure(figsize=(6,4))
        plt.title('Developers by Platform')
        cdict = get_col()
        
        for pl in market.announced_plat:
            temp_dev_rec = [0]*(steps-len(market.dev_rec[str(pl)]))+market.dev_rec[str(pl)]
            market.experiment['developers'][pl] = temp_dev_rec
            labelthis = str(names[str(pl)[0]])+str(gens[str(pl)[-1]])
            plt.plot(temp_dev_rec,linestyle=line[pl],color=cdict[str(pl)[0]],linewidth=1.5,alpha=0.6,label=labelthis)
        plt.ylim(0,d)
        plt.ylabel('Developers')
        plt.xlabel('Steps')
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
        test['dev_plats'] = [[0]*(steps-len(market.dev_rec[str(pl)]))+market.dev_rec[str(pl)] for pl in market.announced_plat]
        
    #Total games per platfor
        plt.figure(figsize=(6,4))
        plt.title('Releases by Platform')
        cdict = get_col()
        
        rel_games = []
        labels = []
        for pl in market.plats:
            rel_games.append(len(pl.games))
            labels.append(str(names[str(pl.ID)[0]])+str(gens[str(pl.ID)[-1]]))
        rel_games = [float(i)/max(rel_games) for i in rel_games]
        rel_games.sort(reverse=True)
        par44 = [float(i)/max(par4) for i in par4]
        plt.bar(np.arange(0,len(rel_games)),rel_games,alpha=0.6,width=1)
        plt.plot(par44,alpha=0.8,label='Emp.')
        plt.ylabel('Releases')
        plt.xlabel('Rank')
        plt.xticks()
        plt.legend(title='Platforms',loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        
        test['releases'] = rel_games
        
    #Distribution of games
        plt.figure(figsize=(6,4))
        plt.title('Developer total unit sales.')
        a = [sum(i.unit_post10weeks.values()) for i in market.devs]
        a.sort(reverse=True)
        market.experiment['dev_rank'] = a
        aa = [float(ii)/max(a) for ii in a]
        ppar3 = [float(ii)/max(par3) for ii in par3]
        plt.plot(np.arange(0,len(aa)),aa,label='Sim.')
        plt.plot(np.arange(0,len(ppar3[:len(aa)])),ppar3[:len(aa)],label='Emp.')
        plt.xlabel('Rank order of developer sales')
        plt.ylabel('Units')
        plt.ylim(min(aa),max(aa))
        #plt.loglog()
        plt.show()
        
        test['dev_rank'] = aa
    
        mean,stdev = get_weekly_games(plot_this=True)
        market.experiment['weekly_mean'] = mean
        market.experiment['weekly_std'] = stdev
        return test

#Export NX to CSV
def export_nx_csv(market):
    nodes = {}
    G = nx.Graph()
    for c in market.cons:
        for splat in c.plat:
            if splat == None:
                nodes[str(c.ID)] = 'None'
            else:
                nodes[str(c.ID)] = splat
        G.add_node(c.ID)
        for e in c.neighbors:
            G.add_edge(c.ID,e)
    edges,index_ = {},0
    for e in G.edges():
        edges[index_] = [e[0],e[1]]
        index_ += 1
    nodedf = pd.DataFrame.from_dict(nodes,orient='index')
    edgedf = pd.DataFrame.from_dict(edges,orient='index')
    nodedf.to_csv('nodes.csv')
    edgedf.to_csv('edges.csv')
    
#Get network of consumers to plot.
def get_network(plot=False,snaps=None):
    if snaps == None:
        G = nx.Graph()
        for c in market.cons:
            G.add_node(c.ID,size=c.plat) #TTRY WITTH ADD COLOR
            for e in c.neighbors:
                G.add_edge(c.ID,e)    
            labels=nx.get_node_attributes(G, 'size')
               
        if plot == True:
           # sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
            plt.grid(False)
            plats = [None,'0_0','1_0','2_0','3_0','4_0']
            colors = {None:'gray','0_0':'y','1_0':'r','2_0':'b','3_0':'#7FFF00','4_0':'purple'}
            pos=nx.spring_layout(G,scale=60)
            for p in plats:
                newlbl = {k:v for k,v in labels.items() if v == p}
                Gplat = G.subgraph(newlbl.keys())
                nx.draw_networkx(Gplat,pos,labels=newlbl,node_size=12,font_size=5,width=0.06,node_color=colors[p],edge_color='black')
            #nx.draw(G,pos,font_size=8,width=0.01,node_size=1)
            lbl = Counter(labels.values())
            print lbl.items()
            plt.show()
        return G

    steps = len(market.cons[0].memory)

    got_pos = False
    cids = [c.ID for c in market.cons]
    plot_sample = rd.sample(cids,len(cids)/2)
    
    for s in [int(i) for i in range(0,steps,steps/snaps)]:
        G = nx.Graph()
        for c in market.cons:
            if c.ID in plot_sample:
                ###
                #TTRY WITTH ADD COLOR
                G.add_node(c.ID,size=c.plat_steps[s])
                
                for e in c.neighbors:
                    G.add_edge(c.ID,e)
                    
                labels=nx.get_node_attributes(G, 'size')
            
            
        if plot == True:
            sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
            plt.grid(False)
            plats = [None,'0_0','1_0','2_0','3_0','4_0']
            colors = {None:'gray','0_0':'y','1_0':'r','2_0':'b','3_0':'#7FFF00','4_0':'purple'}
            if got_pos == False:
                pos=nx.spring_layout(G,scale=60)
                got_pos = True
            for p in plats:
                newlbl = {k:v for k,v in labels.items() if v == p}
                Gplat = G.subgraph(newlbl.keys())
                nx.draw_networkx(Gplat,pos,labels=newlbl,node_size=8,font_size=0,width=0.03,node_color=colors[p],edge_color='white')
            #nx.draw(G,pos,font_size=8,width=0.01,node_size=1)
            lbl = Counter(labels.values())
            plt.title('Step '+str(s+1)+' . Plat. q: '+str(lbl.items()),color='white')
            plt.show()


def animate(p,got_pos = False):
        G = nx.Graph()
        plot_sample = ['missing']
        for c in market.cons:
            if c.ID in plot_sample:
                G.add_node(c.ID,size=c.memory[s])          
                for e in c.neighbors:
                    G.add_edge(c.ID,e)
                    
                labels=nx.get_node_attributes(G, 'size')
            
        plt.grid(False)
        plats = [None,'0_0','1_0','2_0','3_0','4_0']
        colors = {None:'gray','0_0':'y','1_0':'r','2_0':'b','3_0':'#7FFF00','4_0':'purple'}
        if got_pos == False:
            pos=nx.spring_layout(G,scale=60)
            got_pos = True
        for p in plats:
            newlbl = {k:v for k,v in labels.items() if v == p}
            Gplat = G.subgraph(newlbl.keys())
            nx.draw_networkx(Gplat,pos,labels=newlbl,node_size=8,font_size=0,width=0.03,node_color=colors[p],edge_color='white')
        #nx.draw(G,pos,font_size=8,width=0.01,node_size=1)
        lbl = Counter(labels.values())
        return lbl

def get_alert(fail=False):
    duration = 100  # millisecond
    for i in [400,400,200,500,600,700]:
        if fail: i = 700
        winsound.Beep(i, duration)
        if i == 700:
            winsound.Beep(i, 400)

def get_col():
        c = iter(['b','r','g','y','purple'])
        col = {str(i.ID):c.next() for i in market.manuf}
        col['M'] = 'grey'
        col['M_SD'] = 'dark grey'
        return col

def central_weekly(market,report=False,experiment=False):
    all_weeks, dev_rank = [], {}
    for d in market.devs:
        for g in d.unit_games_ts.values():
            all_weeks.append(weekly_gamu(g))
    longest = max([len(g) for g in all_weeks])
    
    all_weeks,ts_weeks = [],[]
    for d in market.devs:
        units = 0
        dev_rank[d.ID] = {'units':0,'games':0,'weekly':[],'release':[]}
        for k,g in d.unit_games_ts.items():
            units += sum(g)
            weekly_ = weekly_gamu(g)
            ts_weeks.append(weekly_)
            weekly_ = weekly_+([0]*(longest-len(weekly_)))
            all_weeks.append(weekly_)
            dev_rank[d.ID]['release'].append(d.unit_games_time[k])
            dev_rank[d.ID]['weekly'].append(weekly_)
        dev_rank[d.ID]['units'] = units
        dev_rank[d.ID]['games'] = len(dev_rank[d.ID]['weekly'])

    avg,std_p,std_n = [],[],[]

    for i in range(longest):
        avg_ = np.average([g[i] for g in all_weeks])
        avg.append(avg_)
        std_ = np.std([g[i] for g in all_weeks])
        std_p.append(avg_+std_)
        std_n.append(avg_-std_)
    sum_g = len(all_weeks)
    sum_g_units = sum([sum(g) for g in all_weeks])
    devdf = pd.DataFrame.from_dict(dev_rank,orient='index')
    devdf = devdf.sort_values(by='units',ascending=False)
    weekly_games = [sum_g,sum_g_units,avg,std_p,std_n,devdf]
    setattr(market,'WeeklyGames',weekly_games)
    
    if experiment == True:
        market.experiment['totals'] = {'plat_units':{}}
        for pl in market.plats:
            market.experiment['totals']['plat_units'][pl] = pl.hwsales[-1]
        market.experiment['totals']['games_out'] = weekly_games[0]
        market.experiment['totals']['unit_sold'] = weekly_games[1]
        market.experiment['totals']['weeks_avg'] = weekly_games[2]
        market.experiment['totals']['week_stdp'] = weekly_games[3]
        market.experiment['totals']['week_stdn'] = weekly_games[4]
        return market
    
    if report:
        print 'Platform units: '
        for p in market.plats:
            print 'Platform '+str(p.ID)+' sold '+str(p.hwsales[-1])
        print ''
        print 'Total games out : '+str(weekly_games[0])
        print 'Total units sold: '+str(weekly_games[1])
        print 'Avg. units sold : '+str(float(weekly_games[1])/weekly_games[0])
        for i in weekly_games[2:5]:
            plt.plot(i)
        plt.title('Average unit sales per week after release.')
        plt.show()
        #best performer time series
        top = devdf.values[0]
        for g in range(len(top[3])): #top[2] = weekly
            plt.plot(top[1][g]*[0]+top[3][g])
        plt.title('Top (percentil 1) performer Units TS')
        plt.show()
        #get p15 performer
        top = devdf.values[len(devdf)*.15]
        for g in range(len(top[3])): #top[2] = weekly
            plt.plot(top[1][g]*[0]+top[3][g])
        plt.title('Percentile 15 performer Units TS')
        plt.show()
        #get worst
        top = devdf.values[len(devdf)*.30]
        for g in range(len(top[3])): #top[2] = weekly
            plt.plot(top[1][g]*[0]+top[3][g])
        plt.title('Percentile 30 performer Units TS')
        plt.show()
        
    return market

def weekly_gamu(g):
    try:
        lg = [g[0]] + [g[gg]-g[gg-1] for gg in range(1,len(g))]
    except:
        lg = [0]
        print 'fail'
    return lg

def post_evaluation(spreferences,product_features):        
        normalized_feat = product_features / np.linalg.norm(product_features)
        normalized_pref = spreferences / np.linalg.norm(spreferences)
        return np.dot(normalized_feat,normalized_pref)
    
def get_experience_distro():
     game_features = [market.games_features[game] for game in market.games_features.keys()]
     plat_features = [p.product for p in market.plats]
     preferences = [c.preferences for c in market.cons]
     plt.hist([post_evaluation(i,plat_features[0]) for i in preferences],bins=50)
     plt.show()
     
def get_degree_distribution():
    directed = [len(c.neighbors) for c in market.cons]
    directed = sorted(directed,reverse=True)
    x,y = np.histogram(directed)
    plt.bar(y[:-1],x,width=1)
    plt.show()

def get_weekly_games(plot_this=False):
    weekly = []
    for dev in market.devs:
        for gamevalues in dev.unit_games_ts.values():
            noncumulative = [gamevalues[i]-gamevalues[i-1] for i in range(1,len(gamevalues))]
            if len(noncumulative) > 11:
                weekly.append(np.array(noncumulative[:11]))
    weekly_ar = np.array(weekly)
    mean = np.mean(weekly_ar,axis=0)
    stdev = np.std(weekly_ar,axis=0)

    if plot_this:
        plt.title('Weekly sales')
        plt.plot(mean,label='Mean')
        plt.plot(mean+stdev,'--',label='Std. Dev')
        plt.plot(mean-stdev,'--')
        plt.legend(loc=2)
        plt.xlabel('Weeks since release')
        plt.ylabel('Units')
        plt.show()
    return mean,stdev

#for sim_i in [Basic1_1,Basic1_2]:#,Basic1_3,Basic1_4,Basic2_1,Basic2_2,Basic2_3,Basic2_4]:
#sim =  G(case=Basic1_1,experiments=1,plot_all=True)