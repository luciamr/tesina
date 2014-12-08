import os
from os.path import exists, isdir, basename, join, splitext


def read_tt_sets_file(filename='sets.txt'):                                                                                                                                         
                                                                                                                                                                                        
    if exists(filename) != False | os.path.getsize(filename) == 0:                                                                                                                                       
        raise IOError("Wrong file path or file empty: "+ filename)                                                                                                                                        
                                                                                                         
        with open(filename, 'rb') as f:                                                                                                                                                                   
            locs, descriptors = cPickle.load(f)                                                                                                                                                           
    else:                                                                                                                                                                                                
        f = open(filename, 'r')                                                                                                                                                                           
        header = f.readline().split()                                                                                                                                                                     
                                                                                                                                                                                                          
        num = int(header[0])  # the number of features                                                                                                                                                    
        featlength = int(header[1])  # the length of the descriptor                                                                                                                                       
        if featlength != 128:  # should be 128 in this case                                                                                                                                               
            raise RuntimeError('Keypoint descriptor length invalid (should be 128).')                                                                                                                     
                                                                                                                                                                                                          
        locs = zeros((num, 4))                                                                                                                                                                            
        descriptors = zeros((num, featlength));                                                                                                                                                           
                                                                                                                                                                                                          
        #parse the .key file                                                                                                                                                                              
        e = f.read().split()  # split the rest into individual elements                                                                                                                                   
        pos = 0                                                                                                                                                                                           
        for point in range(num):                                                                                                                                                                          
            #row, col, scale, orientation of each feature                                                                                                                                                 
            for i in range(4):                                                                                                                                                                            
                locs[point, i] = float(e[pos + i])                                                                                                                                                        
            pos += 4                                                                                                                                                                                      
                                                                                                                                                                                                          
            #the descriptor values of each feature                                                                                                                                                        
            for i in range(featlength):                                                                                                                                                                   
                descriptors[point, i] = int(e[pos + i])                                                                                                                                                   
            #print descriptors[point]                                                                                                                                                                    
            pos += 128                                                                                                                                                                                    
                                                                                                                                                                                                          
            #normalize each input vector to unit length                                                                                                                                                   
            descriptors[point] = descriptors[point] / linalg.norm(descriptors[point])                                                                                                                     
            #print descriptors[point]                                                                                                                                                                     
                                                                                                                                                                                                          
        f.close()                                     
                                                                                                                                                                                                          
    return locs, descriptors    
