import numpy as np
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
import seaborn as sn

def linear_regression(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)

mode = 'u'
# allMat = []
startIdx = 0
plt.figure(figsize=(55,20))
data = np.load('D:\\Myown\\WritePaper\\SemiSupervisedWindDownscaling\\init\\Image\\SourceNC\\std_%s.npy'%mode)
for month in range(12):
    for hour in ['00', '12']:
        date = '2016%02d15%s'%(month+1, hour)
        print(datetime.datetime.strptime(date, '%Y%m%d%H').strftime('%Y-%m-%d %H:%M:%S'))

        '''---------------generate npy--------------'''
        # data = nc.Dataset('D:\\Myown\\WritePaper\\SemiSupervisedWindDownscaling\\init\\Image\\SourceNC\\dem_CZ.nc')
        # dem = data['dem'][:].data[3:,:499]
        # data.close()
        # data = nc.Dataset('D:\\Myown\\WritePaper\\SemiSupervisedWindDownscaling\\init\\Image\\SourceNC\\MSP1_PMSC_AIWSRPF_LAPS-1H-0p01_L88_CZ_%s00_00000-00000.nc'%date)
        # temp = data['%s-component_of_wind_height_above_ground'%mode][:].data[0,2:,1:500]
        # data.close()
        #
        # demStd = []
        # tempStd = []
        #
        # for i in range(dem.shape[0]//3):
        #     for j in range(dem.shape[1]//3):
        #         demPart = dem[i*3:i*3+4,j*3:j*3+4]
        #         tempPart = temp[i*3:i*3+4,j*3:j*3+4]
        #         demStd.append(demPart.std())
        #         tempStd.append(tempPart.std())
        #
        # np.random.seed(2020)
        # np.random.shuffle(demStd)
        # np.random.seed(2020)
        # np.random.shuffle(tempStd)
        #
        # allMat.append(np.array(demStd[::500]))
        # allMat.append(np.array(tempStd[::500]))

        '''---------------plot image--------------'''
        demStd = data[2*startIdx]
        tempStd = data[2*startIdx+1]
        a0, a1 = linear_regression(demStd, tempStd)
        x = [0, max(demStd)]
        y = [a0+a1*xx for xx in x]
        ymax = [yy+max(tempStd)*0.1 for yy in y]
        ymin = [yy-max(tempStd)*0.1 for yy in y]
        plt.subplot(3,8,startIdx+1)
        plt.scatter(demStd, tempStd)
        plt.xlabel('Elevation Standard Deviation')
        plt.ylabel('%s Wind Speed Standard Deviation'%(mode.upper()))
        plt.title(datetime.datetime.strptime(date, '%Y%m%d%H').strftime('%Y-%m-%d %H:%M:%S'))
        plt.plot(x,y, color='red')
        plt.plot(x,ymax, color='red', linestyle="--")
        plt.plot(x,ymin, color='red', linestyle="--")
        plt.fill_between(x,ymin,ymax, color='red',  alpha=.25)
        startIdx += 1
plt.savefig('D:\\Myown\\WritePaper\\SemiSupervisedWindDownscaling\\init\\Image\\%sstd.pdf'%mode.upper())
# plt.show()
print('done')
# temp = np.array(allMat)
# np.save('D:\\Myown\\WritePaper\\SemiSupervisedWindDownscaling\\init\\Image\\SourceNC\\std_%s.npy'%mode, temp)