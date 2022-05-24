#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:12:49 2020

@author: kapoorlab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:25:10 2020

@author: Varun Kapoor
"""
import argparse

class MidogConfig(argparse.Namespace):
    
    def __init__(self, cell_position_name, cell_position_label,  **kwargs):
        
        
           self.cell_position_name = cell_position_name
           self.cell_position_label = cell_position_label
           assert len(cell_position_name) == len(cell_position_label)
    

    def to_json(self):

        
         configCord = {}
       
         for i in range(0, len(self.cell_position_name)):
            
             configCord[self.cell_position_name[i]] = self.cell_position_label[i]
        
         return configCord
         
         
        
          
   

        
