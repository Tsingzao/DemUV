from datetime import datetime
from dateutil.relativedelta import relativedelta
from lib.config import edict as edict
import os
from multiprocessing import Pool
import netCDF4 as nc
import numpy as np


def NC2NPY(input):
    '''(33.0,90.01) - (28.0,95.01) ------------  499 * 499'''
    presentTime = input
    presentStr = presentTime.strftime('%Y%m%d%H')
    yearPath, dayPath, hourPath = presentStr[:4], presentStr[:8], presentStr[:10]
    NCPath = edict.data.LAPSFormat%(yearPath, dayPath, hourPath)
    if not os.path.exists(NCPath):
        print('%s not Exist!'%NCPath)
        return
    data = nc.Dataset(NCPath)
    u = data['u-component_of_wind_height_above_ground'][:].data
    v = data['v-component_of_wind_height_above_ground'][:].data
    data.close()
    u = u[:,2:,1:500]
    v = v[:,2:,1:500]
    data = np.concatenate([u,v], axis=0)
    NPYPath = edict.data.LAPSNPY%hourPath
    os.makedirs(os.path.dirname(NPYPath), exist_ok=True)
    np.save(NPYPath, np.around(data, decimals=2))
    print('Save %s'%NPYPath)
    return



def GRAPESMESO2NPY(input):
    '''(33.01,90.01) - (28.00,94.99) ------------  167 * 167'''
    presentTime = input
    presentStr = presentTime.strftime('%Y%m%d%H')
    yearPath, dayPath, hourPath = presentStr[:4], presentStr[:8], presentStr[:10]
    temp = []
    for mode in ['u', 'v']:
        NCPath = edict.data.GRAPESMESOFormat%(yearPath, dayPath, mode.upper(), hourPath)
        if not os.path.exists(NCPath):
            print('%s not Exist!'%NCPath)
            return
        data = nc.Dataset(NCPath)
        ret = data[edict.data.GRAPESMESOVariable%mode][:].data
        data.close()
        temp.append(ret[:1,1:,:167])
    temp = np.concatenate(temp, axis=0)

    NPYPath = edict.data.GRAPESMESONPY%hourPath
    os.makedirs(os.path.dirname(NPYPath), exist_ok=True)
    np.save(NPYPath, np.around(temp, decimals=2))
    print('Save %s'%NPYPath)
    return


def saveDem():
    data = nc.Dataset('./dataset/DemFeature/dem.nc')
    dem = data['dem'][:].data
    data.close()
    dem = dem[3:, :499]
    dem = np.expand_dims(dem, axis=0)
    os.makedirs(os.path.dirname('./dataset/DemFeature/dem.npy'), exist_ok=True)
    np.save('./dataset/DemFeature/dem.npy', dem)


if __name__ == '__main__':
    # a = saveDem()

    startTime = datetime.strptime('2017070100', '%Y%m%d%H')
    endTime = datetime.strptime('2018010100', '%Y%m%d%H')
    days = (endTime-startTime).days
    timeList = [startTime + relativedelta(hours=hour) for hour in range(days*24)]
    # b = NC2NPY(timeList[0])
    with Pool(20) as p:
        p.map(NC2NPY, timeList)


    startTime = datetime.strptime('2019070100', '%Y%m%d%H')
    endTime = datetime.strptime('2020010100', '%Y%m%d%H')
    days = (endTime-startTime).days
    timeList = [startTime + relativedelta(hours=hour*12) for hour in range(days*2)]
    # c = GRAPESMESO2NPY(timeList[0])
    with Pool(20) as p:
        p.map(GRAPESMESO2NPY, timeList)
    # print(0)
