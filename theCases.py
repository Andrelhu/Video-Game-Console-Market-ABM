# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:38:53 2018

@author: Gaijin
"""

#Case settings.    
Basicsim = {'name' : 'Basic',
         'plats': 2,
         'devs' : 150,
         'cons' : 1000,
         'step' : 250,
         'gens' : 1,
         'dates': [55,175], #sin Pong(75)
         'all'  : True,
         'soc'  : 1,
         'media': 1,
         'bound': 1}

Atari = {'name' : 'Atari_case',
         'plats': 4,
         'devs' : 150,
         'cons' : 1000,
         'step' : 750,
         'gens' : 1,
         'dates': [2*52,7*52,10*52,12*52], #sin Pong(75)
         'all'  : True,
         'soc'  : 1,
         'bound': 1}

#Basic1 = One Platform, 50 devs, 5000 cons.
Basic1_1 = {'name': 'Basic1_HighAll',
         'plats': 1,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55], #sin Pong(75)
         'all'  : True,
         'soc'  : 1,
         'media': 1,
         'bound': 1}

Basic1_2 = {'name': 'Basic1_LowSoc',
         'plats': 1,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55], #sin Pong(75)
         'all'  : True,
         'soc'  : 0.5,
         'media': 1,
         'bound': 1}

Basic1_3 = {'name': 'Basic1_LowBound',
         'plats': 1,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55], #sin Pong(75)
         'all'  : True,
         'soc'  : 1,
         'bound': 0.1}

Basic1_4 = {'name': 'Basic1_LowAll',
         'plats': 1,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55], #sin Pong(75)
         'all'  : True,
         'soc'  : 0.1,
         'bound': 0.1}

#Basic2 = Two Platform, 50 devs, 5000 cons.
Basic2_1 = {'name': 'Basic2_HighAll',
         'plats': 2,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55,110], #sin Pong(75)
         'all'  : True,
         'soc'  : 1,
         'bound': 1}

Basic2_2 = {'name': 'Basic2_LowSoc',
         'plats': 2,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55,110], #sin Pong(75)
         'all'  : True,
         'soc'  : 0.1,
         'bound': 1}

Basic2_3 = {'name': 'Basic2_LowBound',
         'plats': 2,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55,110], #sin Pong(75)
         'all'  : True,
         'soc'  : 1,
         'bound': 0.1}

Basic2_4 = {'name': 'Basic2_LowAll',
         'plats': 2,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55,110], #sin Pong(75)
         'all'  : True,
         'soc'  : 0.1,
         'bound': 0.1}

#Basic3 = One Platform, 50 devs, 5000 cons.
Basic3_1 = {'name': 'Basic3_HighAll',
         'plats': 3,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55,110,165], #sin Pong(75)
         'all'  : True,
         'soc'  : 1,
         'bound': 1}

Basic3_4 = {'name': 'Basic3_LowAll',
         'plats': 3,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55,110,165], #sin Pong(75)
         'all'  : True,
         'soc'  : 0.1,
         'bound': 0.1}


#Basic2 = Two Platform, 50 devs, 5000 cons.
SD2_1 = {'name': 'SD2_HighAll',
         'plats': 3,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55,56,57], #sin Pong(75)
         'all'  : True,
         'soc'  : 1,
         'bound': 1}

SD2_4 = {'name': 'SD2_LowAll',
         'plats': 3,
         'devs' : 50,
         'cons' : 1000,
         'step' : 350,
         'gens' : 1,
         'dates': [55,56,57], #sin Pong(75)
         'all'  : True,
         'soc'  : 0.1,
         'bound': 0.1}