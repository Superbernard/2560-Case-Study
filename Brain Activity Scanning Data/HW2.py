import numpy as np
import glob
from copy import deepcopy
from scipy import stats
from sklearn.decomposition import PCA

#set up your data directory
data_dir = r'C:\Users\Bernard\Desktop\big data\HW2\brainD15'

def authors():
    return ['Bernard Chu', 'Jiarou Quan', 'Bowei Wei']
    
    
def compute_fisher(data_mtx):
    '''function to compute fisher's z transformation'''
    corr = np.corrcoef(np.transpose(data_mtx)) 
    f_score = 1/2 * np.log( np.divide(1+corr, 1-corr) ) 
    np.fill_diagonal(f_score, 0)
    return f_score
        

def main(regions = 15, data_folder = data_dir, output_dir = data_dir, var_pctg = 0.95):
    '''function to loop through each txt file, compute required statistics and perform PCA'''
    x_sum = np.zeros(shape=(regions, regions))
    xsq_sum = np.zeros(shape=(regions, regions))
    x_s = []
    count = 0
    
    for file in sorted(glob.glob(data_folder + '\\' +'*.txt')):
        count+=1
        data = np.loadtxt(file)
        fisherz = compute_fisher(data)
        x_zscore = stats.zscore(data, axis=0)
        x_s.append(x_zscore)
        x_sum = np.add(x_sum, fisherz)
        xsq_sum = np.add(xsq_sum, np.square(fisherz))
        if count == 410:
            x_sum_410 = deepcopy(x_sum)
    
    #flatten then reshape and slice the array to rebuild the (410*4800)x15 arrays
    xs = np.array(x_s).flatten().reshape((4800*820, regions))   
    xtrain = xs[:4800*410, :]
    xtest = xs[4800*410 : , :]
    avg = x_sum / count
    var = np.subtract(xsq_sum / count,  np.square(avg))
    avg_train = x_sum_410 / 410
    avg_test = np.subtract(x_sum, x_sum_410) / 410

    np.savetxt(output_dir + '\\' + 'Fn.csv', avg, delimiter=",")
    np.savetxt(output_dir + '\\' + 'Fv.csv', var, delimiter=",")
    np.savetxt(output_dir + '\\' + 'Ftrain.csv', avg_train, delimiter=",")
    np.savetxt(output_dir + '\\' + 'Ftest.csv', avg_test, delimiter=",")
    
    #choose PCA to do matrix factorization
    pca = PCA(var_pctg)
    pcs = pca.fit_transform(xtrain)   #U
    eigen_vec = pca.components_       #G
    #inverse_transform pcs to approximate x, computed as dot product of pcs and eigen vectors
    reconstruction = pca.inverse_transform(pcs)   #UG
    
    Ctrain = np.cov(xtrain.T)
    Ctest = np.cov(xtest.T)
    Cug = np.cov(reconstruction.T)
    dist_UGtest = np.linalg.norm ( np.subtract(Ctest,Cug) )
    dist_traintest = np.linalg.norm( np.subtract(Ctrain, Ctest) )
    
    np.savetxt(output_dir + '\\' + 'U.csv', pcs, delimiter=",")
    np.savetxt(output_dir + '\\' + 'G.csv', eigen_vec, delimiter=",")
    np.savetxt(output_dir + '\\' + 'CUG.csv', Cug, delimiter=",")
    np.savetxt(output_dir + '\\' + 'Ctrain.csv', Ctrain, delimiter=",")
    np.savetxt(output_dir + '\\' + 'Ctest.csv', Ctest, delimiter=",")
    np.savetxt(output_dir + '\\' + 'CUGCtest.csv', dist_UGtest.reshape(1,), delimiter=",")
    np.savetxt(output_dir + '\\' + 'CtrainCtest.csv', dist_traintest.reshape(1,), delimiter=",")
     
output_dir = r'C:\Users\Bernard\Desktop\big data\HW2'
main(output_dir = output_dir)

