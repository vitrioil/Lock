from model import TModel
from sample import Sample
import numpy as np
if __name__ == '__main__':
	sample = Sample(None,None,None,saved=True)
	x_train,y_train,x_test,y_test = sample.load_dataset()
	model = TModel(5511,101,x_train,y_train,x_test,y_test)
	model.load_model()
	print(np.sum(np.argmax(model.Y_train[71],axis=1)))
	model.detect_triggerword("../Data/lockK.wav",True)#Training/raw/train71.wav")
