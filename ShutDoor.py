from functions import *
def train_model():
    featuresShutDoor = extractFeaturesForFolder('./shutthedoor/')
    ShutDoor_gmm = GaussianMixture(n_components=  6, max_iter = 200, covariance_type='tied',n_init =3)
    ShutDoor_gmm.fit(featuresShutDoor)
    pickle.dump(ShutDoor_gmm,open('trained_model_words/ShutDoor.gmm','wb'))

train_model()