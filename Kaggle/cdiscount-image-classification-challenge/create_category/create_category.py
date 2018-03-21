# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:25:10 2017

@author: Ashis Chanda
"""

#reading CSV file, Taking three list for three category
#For each ID, find its label in three category
#save the lists in pickle file format

import csv
import pickle
category1_list = []
category2_list = []
category3_list = []
 
category_dict = dict()          # id (cate1, cate2, cate3)

with open('category_names.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    headerFlag= False
    
    for row in csvReader:
        if headerFlag:  #avoiding header flag
                
                
            #if the item not in list, add it and take it's index
            if row[1] not in category1_list:
                category1_list.append( row[1] )
                
            index1 = category1_list.index( row[1] )    
            
            #******************************************
            if row[2] not in category2_list:
                category2_list.append( row[2] )
                
            index2 = category2_list.index( row[2] )    
            
            #******************************************
            if row[0] not in category3_list:     # row[0] and row[3] are correspondingly same
                category3_list.append( row[0] )
                
            index3 = category3_list.index( row[0] )    
            
            category_dict[ row[0] ] = [ index1, index2, index3]
                 
            
            
            #print(row)
            
        
        headerFlag = True
        
pickle.dump( category_dict , open( "category_dict.p", "wb" ) )        
pickle.dump( category1_list , open( "category1_list.p", "wb" ) )        
pickle.dump( category2_list , open( "category2_list.p", "wb" ) )        
pickle.dump( category3_list , open( "category3_list.p", "wb" ) )        
