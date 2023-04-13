import pywt
import numpy as np


def calculate_energy(n, wpt_data):
    """
    n：bin示小波包的分解层数
    data_map:由小波包的分解按频率顺序构成的数据列表 
    """
    #第n层能量特征
    energy = []
    for data in wpt_data:
        energy.append(pow(np.linalg.norm(data,ord=None),2))
    return np.array(energy)

def calculate_fluctuation(n, wpt_data):
    
    fluctuation_coefficients = []
    for data in wpt_data:
        sum = 0
        for m in range(len(data)-1):
            sum = sum + (data[m+1] - data[m])
        fluctuation_coefficients.append(sum / (len(data)-1))
    return np.array(fluctuation_coefficients)

def calculate_skewness(n, wpt_data):
    
    skewness = []
    
    for data in wpt_data:
        sk = 0
        ave = np.mean(data)
        molecular = 0
        Denominator = 0
        for c in data:
            molecular = molecular + pow(c - ave,3)
            Denominator = Denominator + pow(c - ave, 2)
        molecular = molecular / len(data)
        Denominator = pow(Denominator / len(data), 3/2)
        
        skewness.append(molecular / Denominator)
        
    return np.array(skewness)

def get_wpt_features(data, n):
    """
    data:[None, features, channels]
    n:小波包分解层数
    return: 小波数据[None, features]
    """
    new_data = []
    samples = data.shape[0]
    channels = data.shape[2]
    for sample in data:
        one_sample = []
        for ni in range(channels):
            wpt_data = []  
            
            wp = pywt.WaveletPacket(data=sample[:, ni], wavelet='db1',mode='symmetric',maxlevel=n)
            for i in [node.path for node in wp.get_level(n, 'freq')]:
                wpt_data.append(wp[i].data)
                
            energy = calculate_energy(n, wpt_data)
            fluctuation_coefficient =  calculate_fluctuation(n, wpt_data)
            skewness = calculate_skewness(n, wpt_data)
            
            a = np.hstack((energy, fluctuation_coefficient))
            b = np.hstack((a, skewness))

            one_sample.append(b)
        one_sample = np.array(one_sample).reshape([-1])
        new_data.append(one_sample)
    return np.array(new_data).reshape((samples, -1))


def get_wpt_data(data, n):
    new_data = []
    samples = data.shape[0]
    channels = data.shape[2]
    for sample in data:
        one_sample = []
        for ni in range(channels):
            wpt_data = []  
            
            wp = pywt.WaveletPacket(data=sample[:, ni], wavelet='db1',mode='symmetric',maxlevel=n)
            for i in [node.path for node in wp.get_level(n, 'freq')]:
                wpt_data.append(wp[i].data)
            one_sample.append(wpt_data)
        one_sample = np.transpose(np.array(one_sample), (1,2,0))
        new_data.append(one_sample)
    return np.array(new_data)