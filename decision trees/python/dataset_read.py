# -*- coding: utf-8 -*-

import monkdata as mk

a = mk.monk3

t = 0
f = 0

for i in range(len(a)):
    #print(a[i].positive)
    
    if a[i].positive==True:
        t += 1
        
    else:
        f += 1
        
print("True: ")
print(t)
print("False: ")
print(f)
#print(len(a))
#print(a[0].posititive)
#print("\n\n")
#print(a[0].attribute)
