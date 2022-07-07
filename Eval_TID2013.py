import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import SiameseNet as SiamNet
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


d_jndq = SiamNet.Siamese().cuda()
d_jndq.load_state_dict(torch.load('BestModelParams.pt'))
d_jndq.eval()


tid2013 = pd.read_csv('E:/tid2013/mos_with_names.txt', delimiter=' ', header=None)
tid2013['d-jndq'] = np.zeros(3000)
tid2013 = tid2013.rename(columns={tid2013.columns[0]:'mos', tid2013.columns[1]:'dist_name'})

with torch.no_grad():
    for i in range(tid2013.shape[0]):
        # Get reference and distorted image names from txt file provided by TID-2013 authors.
        ref_name = tid2013['dist_name'][i].split('_')[0] + '.bmp'
        dist_name = tid2013['dist_name'][i]

        # Load pre-processed Achromatic responses.
        ref = loadmat('D:/deep_sim/tid2013/P_maps/' + ref_name + '.mat')['ref_P']
        dist = loadmat('D:/deep_sim/tid2013/P_maps/' + dist_name + '.mat')['test_P']
        ref, dist = np.expand_dims(ref, axis=0), np.expand_dims(dist, axis=0)
        ref, dist = np.expand_dims(ref, axis=0), np.expand_dims(dist, axis=0)

        # Convert to tensor and wrap in Variable.
        ref, dist = torch.from_numpy(ref).type(torch.cuda.FloatTensor), torch.from_numpy(dist).type(torch.cuda.FloatTensor)
        ref, dist = Variable(ref), Variable(dist)
        # Reshape into expected format.
        # ref, dist = ref.permute(0, 1, 3, 2), dist.permute(0, 1, 3, 2)

        a = ref.data.cpu().numpy()[0,0]
        plt.imshow(a)
        plt.show()

        # Input to Siamese model
        out1, out2 = d_jndq(ref, dist)
        # Calculate the Pairwise distance between output feature vectors to get dissimilarity score.
        tid2013.at[i, 'd-jndq'] = F.pairwise_distance(out1, out2).data.cpu().numpy()

        # Print predicted values
        print(dist_name, tid2013['mos'].iloc[i], tid2013['d-jndq'].iloc[i])


# Save acquired scores to a csv file for ease of read.
# Also to a txt file to use with provided TID-2013 evaluation executable files.
# tid2013.to_csv('./D-JNDQ_withfnames.csv')
# np.savetxt('./D-JNDQ.txt', tid2013['d-jndq'].values, fmt='%1.6f')