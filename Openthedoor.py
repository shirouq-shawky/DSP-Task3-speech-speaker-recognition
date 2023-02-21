from functions import *

def train_model():
    featuresCloseTheDoor = extractFeaturesForFolder('./Openthedoor/')
    Openthedoor_gmm = GaussianMixture(n_components=  6, max_iter = 200, covariance_type='tied',n_init =3)
    Openthedoor_gmm.fit(featuresCloseTheDoor)
    pickle.dump(Openthedoor_gmm,open('trained_model_words/Openthedoor.gmm','wb'))


train_model()