
# -*- coding: utf-8 -*-
from math import sqrt
from math import cos
from math import sin
import math
 
 
 
def rad(d):
    return d * math.pi / 180.0
 
def getDistance(lat1, lng1, lat2, lng2):
    EARTH_REDIUS = 1
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(sin(a/2), 2) + cos(radLat1) * cos(radLat2) * math.pow(sin(b/2), 2)))
    s = s * EARTH_REDIUS
    #print("distance=",s)
    return s
    
if __name__ == '__main__':
    lat1= 0
    lng1= 90
    lat2= 0
    lng2= -90
    getDistance(lat1, lng1, lat2, lng2)

