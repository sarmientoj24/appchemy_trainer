######## HELPER FUCNTIONS inside web app backend. Use on webapp ##########

import math
import numpy as np
import operator
# import pandas as pd

# decoder_loaded_model = load_model('decoder.h5')

'''
    Use log-likelihood to compute likelihood of two mu and sigma vectors
'''
def compute_log_likelihood(z_gen, mu_, log_sigma):
    # (zi−μi(x))22σi(x)2
    # Adopted here https://datascience.stackexchange.com/questions/64097/how-to-use-variational-autoencoders-%CE%BC-and-%CF%83-with-user-generated-z/64163#64163
    term_a = (z_gen - mu_) ** 2
    term_b = 2 * (log_sigma ** 2)
    term_c = -1 * (term_a / term_b)

    # −log(2π−−√σi(x))
    term_d = np.log(((2 * np.pi) ** 0.5) * (log_sigma))
    term_e = term_c - term_d

    likelihood = np.sum(term_e)
    return likelihood

'''
    From the paper, there are two ways to find the nearest similar image for retrieval
    (1) Using log likelihood which uses mu and log_sigma
    (2) Using a fixed epsilon parameter for sampling to get the z_value from the given mu and sigma
'''
def find_nearest(z_var_generated):
    # SQL Query get all of those needed and store to dataframe
    # Do Query here
    z_dim = len(z_var_generated)
    z_var_generated = np.array(z_var_generated) # Convert to numpy

    ids_ = sql_df['id'].to_list()
    df = sql_df[['mu_1', 'mu_2', 'mu_3', 'mu_4', 'mu_5', 
                'mu_6', 'mu_7', 'mu_8', 'mu_9', 'mu_10',
                'log_sigma_1', 'log_sigma_2', 'log_sigma_3', 'log_sigma_4', 'log_sigma_5',
                'log_sigma_6', 'log_sigma_7', 'log_sigma_8', 'log_sigma_9', 'log_sigma_10']]
    
    datapoints_encoded_in_db = df.to_numpy()
    rows = datapoints_encoded_in_db.shape[1]

    likelihoods = {}

    counter = 0
    for data in rows:
        mu_vector = data[:z_dim]
        sigma_vector = data[z_dim:z_dim*2]
        log_likelihood = compute_log_likelihood(z_var_generated,
                                                mu_vector, log_sigma_vector)
        likelihoods[id_s[counter]] = log_likelihood
        counter += 1

    # Sort dictionary by likelihood
    sorted_x = sorted(likelihoods.items(), key=operator.itemgetter(1))

    # return top three descending
    return sorted_x[0], sorted_x[1], sorted_x[2]

'''
    The sliders in which a person interacts submits a generated z-vector
    Use trained decoder to decode the z-vector
'''
def decode_img(z_var_generated):
    # decoder_loaded_model = load_model('decoder.h5')
    img = decoder_loaded_model.predict(z_var_generated)
    return img