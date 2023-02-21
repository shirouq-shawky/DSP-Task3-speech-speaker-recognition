from functions import *

def train_model():
    featuresOpenbook = extractFeaturesForFolder('./Openbook/')
    Openbook_gmm = GaussianMixture(n_components=  6, max_iter = 200, covariance_type='tied',n_init =3)
    Openbook_gmm.fit(featuresOpenbook)
    pickle.dump(Openbook_gmm,open('trained_model_words/Openbook.gmm','wb'))


train_model()