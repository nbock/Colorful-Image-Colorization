from skimage.color import rgb2lab
from data_generator import DataHelper
from config import config
import numpy as np
import os
import sklearn.neighbors as nn


quantized_ab = np.load(os.path.join(config.color_data_path, "pts_in_hull.npy")) #313x2 (ab values)
def write_class_probs():
    '''
    For each image and then for each pixel in the image find its class in the gamut of a,b pairs and then keep a count
    of how many pixels are in a class. so baically its calculating class balances over the entire images to get an idea
    about rare pixels. We finally convert the count to probabilities and saved
    '''
    nn_finder = nn.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(quantized_ab)

    helper = DataHelper(config.train_dir)
    total_images=helper.train_iter.n
    total_batches=total_images//config.batch_size
    prior_prob = np.zeros((quantized_ab.shape[0])) #313 class probabilities

    for _ in range(total_batches):
        batch = helper.train_iter.next()
        print(batch.shape," ",helper.train_iter.batch_index)
        ab_batch = rgb2lab(batch)[:, :, :, 1:] #BxHxWx2
        ab_batch_queries = ab_batch.reshape((-1,2)) #BHWx2
        class_indices = nn_finder.kneighbors(ab_batch_queries,return_distance=False) #indices =BHWx1,[0,312]
        indices = np.ravel(class_indices) #BHW
        counts = np.bincount(indices) #313
        prior_prob += counts
    prior_prob /= (1.0 * np.sum(prior_prob))
    np.save(os.path.join(config.color_data_path,"class_probabilities.npy"),prior_prob)

def write_rarity_weights():
    '''
        Now for each class we want a "rarity weightage". Each class in the gamut has a rarity which is kind of inverse
        of its probability. this is calculated exactly as the equation in research paper.
    '''
    prob_class = np.load(os.path.join(config.color_data_path, "class_probabilities.npy"))
    # prob_class = np.load(os.path.join(config.color_data_path, "prior_probs.npy"))
    gamma=0.5 #weight of uniform distribution
    q=prob_class.shape[0]
    prob_uniform=np.zeros(q)
    for i in range(q):
        if prob_class[i]!=0:
            prob_uniform[i]=1
    prob_uniform/=np.sum(1.0*prob_uniform) #uniform distribution

    prob_mix=(1-gamma)*prob_class + gamma*prob_uniform #weighted mixture
    rarity=prob_mix**-1 #rarity is inverse of probability
    prior_factor= rarity/np.sum(rarity*prob_class)
    print("done")
    np.save(os.path.join(config.color_data_path, "prior_factor.npy"), prior_factor)



if __name__ == '__main__':
    write_class_probs()
    write_rarity_weights()



