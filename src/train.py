from model import TModel
from sample import Sample
import numpy as np
if __name__ == '__main__':
	sample = Sample(None,None,None,saved=True)
	x_train,y_train,x_test,y_test = sample.load_dataset()
	model = TModel(5511,101,x_train,y_train,x_test,y_test);model.load_model()
	print(((model.Y_train.shape[0]*1375)-np.sum(np.argmax(model.Y_train,axis=2)))/(model.Y_train.shape[0]*1375))#model.load_model()
	#print(np.sum(np.argmax(model.Y_train[5],axis=1)));print(model.Y_train[5])
	#model.detect_triggerword("../Data/Training/raw/train5.wav")
	model.train(lr=1e-5,epochs=50,batch_size=32,saved=True)
