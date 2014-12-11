import os
from os.path import exists, isdir, basename, join, splitext

def read_tt_sets_file(filename='sets.txt'):
    if exists(filename) != False | os.path.getsize(filename) == 0:
        raise IOError('Wrong file path or file empty: ' + filename)
    else:
        with open(filename, 'rb') as f:
            header = f.readline().split(' ')
            num_cat = int(header[0])
            num_set = int(header[1])
            all_cats = []
            all_files = {}
            for i in range(0, num_cat):
                cat_name = f.readline().rstrip('\n')
                all_cats.append(cat_name)
                cat_path = f.readline().rstrip('\n')
                cat_files = {}
                for j in range(0, num_set):
                    train = f.readline().rstrip('\n').split(' ')
                    test = f.readline().rstrip('\n').split(' ')
                    #train_path = [cat_path + t for t in train]
                    #test_path = [cat_path + t for t in test]
                    cat_files[j] = (train, test)
                all_files[cat_path] = cat_files
            f.close()
        return all_cats, all_files
                        
                                                                                                                                                                                                                    
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
