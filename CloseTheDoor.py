from functions import *

def train_model():
    featuresCloseTheDoor = extractFeaturesForFolder('./CloseTheDoor/')
    CloseTheDoor_gmm = GaussianMixture(n_components=  6, max_iter = 200, covariance_type='tied',n_init =3)
    CloseTheDoor_gmm.fit(featuresCloseTheDoor)
    pickle.dump(CloseTheDoor_gmm,open('trained_model_words/CloseTheDoor.gmm','wb'))


train_model()