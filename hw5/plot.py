from keras.utils.vis_utils import plot_model
from keras.models import load_model
emotion_classifier = load_model('model')
emotion_classifier.summary()
plot_model(emotion_classifier,to_file='model.png')