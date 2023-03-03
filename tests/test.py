import sys, logging
import unittest
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
import torch

from transformers import AutoModel, TFAutoModelForSequenceClassification, AlbertModel

import weightwatcher as ww
from weightwatcher import RMT_Util
from weightwatcher import WW_powerlaw
from weightwatcher.constants import  *


import numpy as np
import pandas as pd
import torchvision.models as models

import gc

warnings.simplefilter(action='ignore', category=RuntimeWarning)

import matplotlib
matplotlib.use('Agg')

class Test_Base(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		"""I run only once for this class
		"""



	@classmethod
	def tearDownClass(cls):
		gc.collect()
		tf.keras.backend.clear_session()
		torch.cuda.empty_cache()


	def tearDown(self):
		if hasattr(self, 'model'  ): del self.model
		if hasattr(self, 'watcher'): del self.watcher
		gc.collect()
		tf.keras.backend.clear_session()
		torch.cuda.empty_cache()
		
class Test_ValidParams(Test_Base):

	def setUp(self):
		"""I run before every test in this class
			
		"""
		print("\n-------------------------------------\nIn Test_ValidParams:", self._testMethodName)

	def test_valid_params(self):
		params = DEFAULT_PARAMS.copy()

		valid = ww.WeightWatcher.valid_params(params)
		self.assertTrue(valid)
		
		params = DEFAULT_PARAMS.copy()		
		params[PL_PACKAGE]=WW_POWERLAW_PACKAGE
		params[FIX_FINGERS]=CLIP_XMAX
		valid = ww.WeightWatcher.valid_params(params)
		self.assertTrue(valid)
		
		params = DEFAULT_PARAMS.copy()
		params[PL_PACKAGE]=WW_POWERLAW_PACKAGE
		params[XMAX]= -1
		valid = ww.WeightWatcher.valid_params(params)
		self.assertTrue(valid)
		
		params = DEFAULT_PARAMS.copy()
		params[PL_PACKAGE]=WW_POWERLAW_PACKAGE
		params[XMAX] = 1
		valid = ww.WeightWatcher.valid_params(params)
		self.assertTrue(valid)
		
		params = DEFAULT_PARAMS.copy()
		params[PL_PACKAGE]=POWERLAW_PACKAGE
		params[XMAX]='force'
		valid = ww.WeightWatcher.valid_params(params)
		self.assertTrue(valid)
		
		
	def test_invalid_PL_package_settings(self):
		params = DEFAULT_PARAMS.copy()
		params[PL_PACKAGE]=WW_POWERLAW_PACKAGE
		params[XMAX]='force'
		valid = ww.WeightWatcher.valid_params(params)
		self.assertFalse(valid)
		
		params = DEFAULT_PARAMS.copy()
		params[PL_PACKAGE]=WW_POWERLAW_PACKAGE
		params[FIT]=TPL
		valid = ww.WeightWatcher.valid_params(params)
		self.assertFalse(valid)
		
		params = DEFAULT_PARAMS.copy()
		params[PL_PACKAGE]=WW_POWERLAW_PACKAGE
		params[FIT]=E_TPL
		valid = ww.WeightWatcher.valid_params(params)
		self.assertFalse(valid)
		
		params = DEFAULT_PARAMS.copy()
		params[PL_PACKAGE]=WW_POWERLAW_PACKAGE
		params[FIT]=TRUNCATED_POWER_LAW
		valid = ww.WeightWatcher.valid_params(params)
		self.assertFalse(valid)
				

		params = DEFAULT_PARAMS.copy()
		params[XMAX]=0
		valid = ww.WeightWatcher.valid_params(params)
		self.assertFalse(valid)

		params = DEFAULT_PARAMS.copy()
		params[FIX_FINGERS]=True
		params[XMAX]=1
		valid = ww.WeightWatcher.valid_params(params)
		self.assertFalse(valid)

class Test_KerasLayers(Test_Base):
	
	
	def setUp(self):
		"""I run before every test in this class
		
			Creats a VGG16 model and gets the last layer,
			
		"""
		print("\n-------------------------------------\nIn Test_KerasLayers:", self._testMethodName)
		self.model = VGG16() 
		self.last_layer = self.model.submodules[-1]
		ww.weightwatcher.keras = keras

		 		
	#TODO:  M
	# test ww_layer matches framework_layer
	# test ww_layer_weights_and_biases
	# test kears_layer_weights_and_biases
	# 
	# think on this...document carefuly
	
	def test_keras_layer_constructor(self):
				
		expected_layer_id = 10
		expected_name = "test_name"
		expected_longname = "test_longname"

		actual_layer = ww.weightwatcher.KerasLayer(self.last_layer, layer_id=expected_layer_id, name=expected_name, longname=expected_longname)

		actual_name = actual_layer.name
		self.assertEqual(expected_name, actual_name)
		
		actual_longname = actual_layer.longname
		self.assertEqual(expected_longname, actual_longname)
		
		self.assertTrue(actual_layer.plot_id is not None)
		self.assertFalse(actual_layer.skipped)
		self.assertEqual(actual_layer.channels, CHANNELS.FIRST)
		self.assertEqual(actual_layer.framework,FRAMEWORK.KERAS)
		self.assertEqual(actual_layer.the_type, LAYER_TYPE.DENSE)
		self.assertTrue(actual_layer.has_biases())

		expected_type = "<class 'weightwatcher.weightwatcher.KerasLayer'>"
		actual_type = str(type(actual_layer))
		self.assertEqual(expected_type, actual_type)
		

		keras_layer = actual_layer.layer
		self.assertEqual(keras_layer, self.last_layer)
		
		expected_type = "<class 'keras.layers.core.dense.Dense'>"
		actual_type = str(type(keras_layer))
		self.assertEqual(expected_type, actual_type)
		

	
	
	def test_ww_layer_iterator(self):
		"""Test that the layer iterators iterates over al layers as expected"""
		
		expected_num_layers = 15 # I think 16 is the flattened layer
		layer_iterator = ww.WeightWatcher().make_layer_iterator(self.model)
		self.assertTrue(layer_iterator is not None)
		num_layers = 0
		for ww_layer in layer_iterator:
			num_layers += 1
		self.assertEqual(expected_num_layers, num_layers)
		
		
		expected_type = "<class 'weightwatcher.weightwatcher.WWLayer'>"
		actual_type = str(type(ww_layer))
		self.assertEqual(expected_type, actual_type)
		
		
	def get_last_layer(self):
		layer_iterator = ww.WeightWatcher().make_layer_iterator(self.model)
		num_layers = 0
		for ww_layer in layer_iterator:
			num_layers += 1
		return ww_layer
	
	def test_ww_layer_attributes(self):
		
		ww_layer = self.get_last_layer()
					
		expected_type = "<class 'weightwatcher.weightwatcher.WWLayer'>"
		actual_type = str(type(ww_layer))
		self.assertEqual(expected_type, actual_type)
		
		expected_name = "predictions"
		actual_name = ww_layer.name
		self.assertEqual(expected_name, actual_name)
		
		framework_layer = ww_layer.framework_layer
		self.assertTrue(framework_layer is not None)
		
		expected_type = "<class 'weightwatcher.weightwatcher.KerasLayer'>"
		actual_type = str(type(framework_layer))
		self.assertEqual(expected_type, actual_type)
	
		self.assertEqual(ww_layer.name, framework_layer.name)
		
		# swhy is longname none ?
		print(f"the longname is {framework_layer.longname}")
		
		has_weights, weights, has_biases, biases  = ww_layer.get_weights_and_biases()
		self.assertTrue(has_weights)
		self.assertTrue(has_biases)
		self.assertTrue(weights is not None)
		self.assertTrue(biases is not None)
		
		expected_W_shape = (4096, 1000)
		expected_B_shape = (1000,)
		actual_W_shape = weights.shape
		actual_B_shape = biases.shape
		
		self.assertEqual(expected_W_shape, actual_W_shape)
		self.assertEqual(expected_B_shape, actual_B_shape)
		


	def test_replace_weights_only(self):
	
		last_layer = self.get_last_layer()
		has_weights, weights, has_biases, biases   = last_layer.get_weights_and_biases()

		expected_old_W_min = np.min(weights)
		expected_old_B_min = np.min(biases)


		new_weights = np.ones_like(weights)	
		
		last_layer.replace_layer_weights(new_weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, 1.0)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)

		# put the weights back
		last_layer.replace_layer_weights(weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.0
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, expected_old_W_min)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)
	
	def test_replace_weights_and_biases(self):
			
		last_layer = self.get_last_layer()
		has_weights, weights, has_biases, biases   = last_layer.get_weights_and_biases()

		expected_old_W_min = np.min(weights)
		expected_old_B_min = np.min(biases)


		new_weights = np.ones_like(weights)	
		new_biases = 2*np.ones_like(biases)	

		
		last_layer.replace_layer_weights(new_weights, new_biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, 1.0)
		self.assertEqual(replaced_new_B_min, 2.0)

		# put the weights back
		last_layer.replace_layer_weights(weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.0
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, expected_old_W_min)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)
		
		

			

class Test_PyTorchLayers(Test_Base):
	
	
	def setUp(self):
		"""I run before every test in this class
		
			Creats a VGG16 model and gets the last layer,
			
		"""
		print("\n-------------------------------------\nIn Test_PyTorchLayers:", self._testMethodName)
		ww.weightwatcher.torch = torch
		self.model = models.resnet18()
		for layer in self.model.modules():
			self.last_layer = layer	
			
	def test_pytorch_layer_constructor(self):
				
		expected_layer_id = 21
		expected_name = "test_name"
		expected_longname = "test_longname"

		actual_layer = ww.weightwatcher.PyTorchLayer(self.last_layer, layer_id=expected_layer_id, name=expected_name, longname=expected_longname)

		actual_name = actual_layer.name
		self.assertEqual(expected_name, actual_name)
		
		actual_longname = actual_layer.longname
		self.assertEqual(expected_longname, actual_longname)
		
		self.assertTrue(actual_layer.plot_id is not None)
		self.assertFalse(actual_layer.skipped)
		self.assertEqual(actual_layer.framework,FRAMEWORK.PYTORCH)
		self.assertEqual(actual_layer.channels, CHANNELS.LAST)
		self.assertEqual(actual_layer.the_type, LAYER_TYPE.DENSE)
		self.assertTrue(actual_layer.has_biases())

		expected_type = "<class 'weightwatcher.weightwatcher.PyTorchLayer'>"
		actual_type = str(type(actual_layer))
		self.assertEqual(expected_type, actual_type)
		

		pytorch_layer = actual_layer.layer
		self.assertEqual(pytorch_layer, self.last_layer)
		
		expected_type = "<class 'torch.nn.modules.linear.Linear'>"
		actual_type = str(type(pytorch_layer))
		self.assertEqual(expected_type, actual_type)
		
	
	
	
	def test_ww_layer_iterator(self):
		"""Test that the layer iterators iterates over al layers as expected"""
		
		expected_num_layers = 21 # I think 16 is the flattened layer
		layer_iterator = ww.WeightWatcher().make_layer_iterator(self.model)
		self.assertTrue(layer_iterator is not None)
		num_layers = 0
		for ww_layer in layer_iterator:
			num_layers += 1
		self.assertEqual(expected_num_layers, num_layers)
		
		
		expected_type = "<class 'weightwatcher.weightwatcher.WWLayer'>"
		actual_type = str(type(ww_layer))
		self.assertEqual(expected_type, actual_type)
		
	def get_last_layer(self):
		layer_iterator = ww.WeightWatcher().make_layer_iterator(self.model)
		num_layers = 0
		for ww_layer in layer_iterator:
			num_layers += 1
		return ww_layer
	
	def test_ww_layer_attributes(self):
		
		ww_layer = self.get_last_layer()
					
		expected_type = "<class 'weightwatcher.weightwatcher.WWLayer'>"
		actual_type = str(type(ww_layer))
		self.assertEqual(expected_type, actual_type)
		
		expected_name = "Linear"
		actual_name = ww_layer.name
		self.assertEqual(expected_name, actual_name)
		
		framework_layer = ww_layer.framework_layer
		self.assertTrue(framework_layer is not None)
		
		expected_type = "<class 'weightwatcher.weightwatcher.PyTorchLayer'>"
		actual_type = str(type(framework_layer))
		self.assertEqual(expected_type, actual_type)
	
		self.assertEqual(ww_layer.name, framework_layer.name)
		
		# swhy is longname none ?
		print(f"the longname is {framework_layer.longname}")
		
		has_weights, weights, has_biases, biases  = ww_layer.get_weights_and_biases()
		self.assertTrue(has_weights)
		self.assertTrue(has_biases)
		self.assertTrue(weights is not None)
		self.assertTrue(biases is not None)
		
		expected_W_shape = (1000, 512)
		expected_B_shape = (1000,)
		actual_W_shape = weights.shape
		actual_B_shape = biases.shape
		
		self.assertEqual(expected_W_shape, actual_W_shape)
		self.assertEqual(expected_B_shape, actual_B_shape)
		


	def test_replace_weights_only(self):
	
		last_layer = self.get_last_layer()
		has_weights, weights, has_biases, biases   = last_layer.get_weights_and_biases()

		expected_old_W_min = np.min(weights)
		expected_old_B_min = np.min(biases)


		new_weights = np.ones_like(weights)	
		
		last_layer.replace_layer_weights(new_weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, 1.0)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)

		# put the weights back
		last_layer.replace_layer_weights(weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.0
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, expected_old_W_min)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)
	
	def test_replace_weights_and_biases(self):
			
		last_layer = self.get_last_layer()
		has_weights, weights, has_biases, biases   = last_layer.get_weights_and_biases()

		expected_old_W_min = np.min(weights)
		expected_old_B_min = np.min(biases)


		new_weights = np.ones_like(weights)	
		new_biases = 2*np.ones_like(biases)	

		
		last_layer.replace_layer_weights(new_weights, new_biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		
		replaced_new_W_min = np.min(replaced_weights) # 1.
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, 1.0)
		self.assertEqual(replaced_new_B_min, 2.0)

		# put the weights back
		last_layer.replace_layer_weights(weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.0
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, expected_old_W_min)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)
		
	
		
	

class Test_PyStateDictLayers(Test_Base):
	
	
	def setUp(self):
		"""I run before every test in this class
		
			Creats a VGG16 model and gets the last layer,
			
		"""
		#import inspect

		print("\n-------------------------------------\nIn Test_PyStateDictLayers:", self._testMethodName)
		ww.weightwatcher.torch = torch
		self.model = models.resnet18().state_dict()
		#for cls in inspect.getmro(type(self.model)):
		#	print(str(cls))
                
		for key in self.model.keys():
			if key.endswith('.weight'):
				layer_name = key[:-len('.weight')]
				self.last_layer_name = layer_name	
			
			
	def test_pytorch_layer_constructor(self):
				
		expected_layer_id = 21
		expected_name = self.last_layer_name 
		expected_longname = expected_name

		actual_layer = ww.weightwatcher.PyStateDictLayer(self.model, expected_layer_id, self.last_layer_name)

		print(actual_layer)
		
		actual_name = actual_layer.name
		self.assertEqual(expected_name, actual_name)
		
		actual_longname = actual_layer.longname
		self.assertEqual(expected_longname, actual_longname)
		
		self.assertTrue(actual_layer.plot_id is not None)
		self.assertFalse(actual_layer.skipped)
		self.assertEqual(actual_layer.channels, CHANNELS.LAST)
		self.assertEqual(actual_layer.framework,FRAMEWORK.PYSTATEDICT)
		self.assertEqual(actual_layer.the_type, LAYER_TYPE.DENSE)
		self.assertTrue(actual_layer.has_biases())

		expected_type = "<class 'weightwatcher.weightwatcher.PyStateDictLayer'>"
		actual_type = str(type(actual_layer))
		self.assertEqual(expected_type, actual_type)
	
	
	
	
	def test_ww_layer_iterator(self):
		"""Test that the layer iterators iterates over al layers as expected"""
		
		expected_num_layers = 21 # I think 16 is the flattened layer
		layer_iterator = ww.WeightWatcher().make_layer_iterator(self.model)
		
		self.assertTrue(layer_iterator is not None)
		num_layers = 0
		for ww_layer in layer_iterator:
			num_layers += 1
		self.assertEqual(expected_num_layers, num_layers)
		
		
		expected_type = "<class 'weightwatcher.weightwatcher.WWLayer'>"
		actual_type = str(type(ww_layer))
		self.assertEqual(expected_type, actual_type)
		
	def get_last_layer(self):
		layer_iterator = ww.WeightWatcher().make_layer_iterator(self.model)
		num_layers = 0
		for ww_layer in layer_iterator:
			num_layers += 1
			print(ww_layer)
		return ww_layer
	
	def test_ww_layer_attributes(self):
		
		ww_layer = self.get_last_layer()
					
		expected_type = "<class 'weightwatcher.weightwatcher.WWLayer'>"
		actual_type = str(type(ww_layer))
		self.assertEqual(expected_type, actual_type)
		
		expected_name = 'fc'
		actual_name = ww_layer.name
		self.assertEqual(expected_name, actual_name)
		
		framework_layer = ww_layer.framework_layer
		self.assertTrue(framework_layer is not None)
		
		expected_type = "<class 'weightwatcher.weightwatcher.PyStateDictLayer'>"
		actual_type = str(type(framework_layer))
		self.assertEqual(expected_type, actual_type)
	
		self.assertEqual(ww_layer.name, framework_layer.name)
		
		# swhy is longname none ?
		print(f"the longname is {framework_layer.longname}")
		
		has_weights, weights, has_biases, biases  = ww_layer.get_weights_and_biases()
		self.assertTrue(has_weights)
		self.assertTrue(has_biases)
		self.assertTrue(weights is not None)
		self.assertTrue(biases is not None)
		
		expected_W_shape = (1000, 512)
		expected_B_shape = (1000,)
		actual_W_shape = weights.shape
		actual_B_shape = biases.shape
		
		self.assertEqual(expected_W_shape, actual_W_shape)
		self.assertEqual(expected_B_shape, actual_B_shape)
		


	def test_replace_weights_only(self):
	
		last_layer = self.get_last_layer()
		has_weights, weights, has_biases, biases   = last_layer.get_weights_and_biases()

		expected_old_W_min = np.min(weights)
		expected_old_B_min = np.min(biases)


		new_weights = np.ones_like(weights)	
		
		last_layer.replace_layer_weights(new_weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, 1.0)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)

		# put the weights back
		last_layer.replace_layer_weights(weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.0
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, expected_old_W_min)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)
	
	def test_replace_weights_and_biases(self):
			
		last_layer = self.get_last_layer()
		has_weights, weights, has_biases, biases   = last_layer.get_weights_and_biases()


		expected_old_W_min = np.min(weights)
		expected_old_B_min = np.min(biases)


		new_weights = np.ones_like(weights)	
		new_biases = 2*np.ones_like(biases)	

		
		last_layer.replace_layer_weights(new_weights, new_biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		
		replaced_new_W_min = np.min(replaced_weights) # 1.
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, 1.0)
		self.assertEqual(replaced_new_B_min, 2.0)

		# put the weights back
		last_layer.replace_layer_weights(weights, biases)
		has_replaced_weights, replaced_weights, has_replaced_biases, replaced_biases   = last_layer.get_weights_and_biases()
		replaced_new_W_min = np.min(replaced_weights) # 1.0
		replaced_new_B_min = np.min(replaced_biases) # 1.0

		self.assertEqual(replaced_new_W_min, expected_old_W_min)
		self.assertEqual(replaced_new_B_min, expected_old_B_min)
		
	
class Test_VGG11_noModel(Test_Base):
	
	"""Same as Test_VGG11 methods, but the model is not specified in setup """

		
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_VGG11_noModel:", self._testMethodName)
		
		self.params = DEFAULT_PARAMS.copy()
		# use older power lae
		self.params[PL_PACKAGE]=POWERLAW
		self.params[XMAX]=XMAX_FORCE
		
		self.model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		self.watcher = ww.WeightWatcher(log_level=logging.WARNING)
		
		self.first_layer = 2
		self.second_layer = 5
		self.third_layer = 8

		self.fc1_layer = 25
		self.fc2_layer = 28
		self.fc3_layer = 31

		self.fc_layers = [self.fc1_layer, self.fc2_layer, self.fc3_layer]
		self.min_layer_id = self.first_layer


	def test_basic_columns_no_model(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
				
		details = self.watcher.describe(model=self.model)
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")
		
		print(details)

		for key in ['layer_id', 'name', 'M', 'N', 'Q', 'longname']:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))

		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		Q = details.Q.to_numpy()[0]

		self.assertAlmostEqual(Q, N/M, places=2)

		
	def test_analyze_columns_no_model(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		
		details = self.watcher.analyze(model=self.model, layers=[self.fc2_layer])
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		columns = "layer_id,name,D,M,N,alpha,alpha_weighted,has_esd,lambda_max,layer_type,log_alpha_norm,log_norm,log_spectral_norm,norm,num_evals,rank_loss,rf,sigma,spectral_norm,stable_rank,sv_max,xmax,xmin,num_pl_spikes,weak_rank_loss,fit_entropy".split(',')
		print(details.columns)
		for key in columns:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))
			
			
					
	def test_svd_smoothing_no_model(self):
		"""Test the svd smoothing on 1 layer of VGG
		"""
		
		# 819 =~ 4096*0.2
		self.watcher.SVDSmoothing(model=self.model, layers=[self.fc2_layer])
		esd = self.watcher.get_ESD(layer=self.fc2_layer) 
		num_comps = len(esd[esd>10**-10])
		self.assertEqual(num_comps, 819)
		
		
		
				
	def test_get_summary_no_model(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		
		description = self.watcher.describe(model=self.model)
		self.assertEqual(11, len(description))
		
		
		details = self.watcher.analyze(model=self.model, layers=[self.fc2_layer])
		returned_summary = self.watcher.get_summary(details)
		
		print(returned_summary)
		
		saved_summary = self.watcher.get_summary()
		self.assertEqual(returned_summary, saved_summary)
		
		
				
	def test_svd_sharpness_no_model(self):
		"""Test the svd smoothing on 1 layer of VGG
		"""
 		
	
		esd_before = self.watcher.get_ESD(model=self.model, layer=self.fc2_layer) 
		
		self.watcher.SVDSharpness(model=self.model, layers=[self.fc2_layer])
		esd_after = self.watcher.get_ESD(layer=self.fc2_layer) 
		
		print("max esd before {}".format(np.max(esd_before)))
		print("max esd after {}".format(np.max(esd_after)))

		self.assertGreater(np.max(esd_before)-2.0,np.max(esd_after))
		
		
		
	def test_getESD_no_model(self):
		"""Test that eigenvalues are available while specifying the model explicitly
		"""

		esd = self.watcher.get_ESD(model=self.model, layer=self.second_layer)
		self.assertEqual(len(esd), 576)
		
		
						
	def test_permute_W_no_model(self):
		"""Test that permute and unpermute methods work
		"""
		N, M = 4096, 4096
		iterator = self.watcher.make_layer_iterator(model=self.model, layers=[self.fc2_layer])
		for ww_layer in iterator:
			self.assertEqual(ww_layer.layer_id,self.fc2_layer)
			W = ww_layer.Wmats[0]
			self.assertEqual(W.shape,(N,M))
			
			self.watcher.apply_permute_W(ww_layer)
			W2 = ww_layer.Wmats[0]
			self.assertNotEqual(W[0,0],W2[0,0])
			
			self.watcher.apply_unpermute_W(ww_layer)
			W2 = ww_layer.Wmats[0]
			self.assertEqual(W2.shape,(N,M))
			self.assertEqual(W[0,0],W2[0,0])
			
			
			
	def test_randomize_no_model(self):
		"""Test randomize option : only checks that the right columns are present, not the values
		"""
		
		rand_columns = ['max_rand_eval', 'rand_W_scale', 'rand_bulk_max',
					 'rand_bulk_min', 'rand_distance', 'rand_mp_softrank', 
					 'rand_num_spikes', 'rand_sigma_mp']
       
		details = self.watcher.analyze(model=self.model, layers = self.fc2_layer, randomize=False)	
		for column in rand_columns:
			self.assertNotIn(column, details.columns)
			
		details = self.watcher.analyze(layers = [self.fc2_layer], randomize=True)	
		for column in rand_columns:	
			self.assertIn(column, details.columns)
			
			
	def test_intra_power_law_fit_no_model(self):
		"""Test PL fits on intra
		"""

		details= self.watcher.analyze(model=self.model, layers=self.fc_layers, intra=True, randomize=False, vectors=False)
		actual_alpha = details.alpha[0]
		actual_best_fit = details.best_fit[0]
		print(actual_alpha,actual_best_fit)

		expected_alpha =  2.654 # not very accurate because of the sparisify transform
		expected_best_fit = LOG_NORMAL
		self.assertAlmostEqual(actual_alpha,expected_alpha, places=1)
		self.assertEqual(actual_best_fit, expected_best_fit)

			


class Test_VGG11_Distances(Test_Base):

	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_VGG11_Distances:", self._testMethodName)
		self.model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)
		
		self.first_layer = 2
		self.second_layer = 5
		self.third_layer = 8

		self.fc1_layer = 25
		self.fc2_layer = 28
		self.fc3_layer = 31
		
		self.fc_layers = [self.fc1_layer, self.fc2_layer, self.fc3_layer]
		self.min_layer_id = self.first_layer
		
		return
	
	
		
	def test_same_distances(self):
		"""Test that the distance method works correctly between the same model
        """
        
		m1 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		m2 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		avg_dW, avg_db, distances = self.watcher.distances(m1, m2)
		
		actual_mean_distance = avg_dW
		expected_mean_distance = 0.0	                       
		self.assertEqual(actual_mean_distance,expected_mean_distance)
		
		actual_mean_distance = avg_db
		expected_mean_distance = 0.0	                       
		self.assertEqual(actual_mean_distance,expected_mean_distance)
		
		print(distances)

	def test_distances(self):
		"""Test that the distance method works correctly between different model
                """
		m1 = models.vgg11()
		m2 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		avg_dW, avg_db, distances = self.watcher.distances(m1, m2)
	
		print(avg_dW,avg_db)
		actual_mean_distance = avg_dW
		expected_mean_distance = 46.485
		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)
		
		actual_mean_distance = avg_db
		expected_mean_distance = 0.67622
		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)
		
		print(distances)

		

	def test_Euclidian_distances(self):
		"""Test that the distance method works correctly when method='EUCLIDEAN'
        """
                
		m1 = models.vgg11()
		m2 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		avg_dW, avg_db, distances = self.watcher.distances(m1, m2, method=EUCLIDEAN)
		
		actual_mean_distance = avg_dW
		expected_mean_distance = 30.2
		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)
		
		# biased not implemented yet in layers
		actual_mean_distance = avg_db
		expected_mean_distance = 0.00
		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)
		


	def test_Euclidian_distances_w_one_layer(self):
		"""Test that the distance method works correctly when methdod='RAW', 1 layer
                """
		m1 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		m2 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		avg_dW, avg_db, distances = self.watcher.distances(m1, m2, method=EUCLIDEAN, layers=[self.fc2_layer])
		actual_mean_distance = avg_dW
		expected_mean_distance = 0.0

		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)
		# TODO: test length of distances also

	# TODO implement with centering
	def test_CKA_distances(self):
		"""Test that the distance method works correctly for CKA method,  ww2x=False | pool=True
               
            Note: biases are not treated yyet
        """
		m1 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		m2 = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		avg_dW, avg_db, distances =  self.watcher.distances(m1, m2, method=CKA, pool=True)
		
		print("====== pool=True ========")
		print(distances)
		
		actual_mean_distance = avg_dW
		expected_mean_distance = 1.0
		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)
		
		actual_mean_distance = avg_db
		expected_mean_distance = 0.0
		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)


		avg_dW, avg_db, distances = self.watcher.distances(m1, m2, method=CKA, pool=False)
				
		print("====== pool=False ========")
		print(distances)
		
		actual_mean_distance = avg_dW
		expected_mean_distance = 1.0
		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)
		
		actual_mean_distance = avg_db
		expected_mean_distance = 0.0
		self.assertAlmostEqual(actual_mean_distance,expected_mean_distance, places=1)
		
		

class Test_Albert(Test_Base):
	"""Test using the xxlarge ALBERT model
	
	WarningL: this test requires downloading a  large model
	
	This exists nmostly to test the fix fingers option an an important model
	
	"""

	def setUp(self):

		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_Albert:", self._testMethodName)
		
		self.params = DEFAULT_PARAMS.copy()
		model_name = f"albert-xxlarge-v2"
		self.model  = AlbertModel.from_pretrained(model_name)
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)
		
			
		
	def test_albert_availble(self):
		
		details = self.watcher.describe(layers=[17])
		print(details)
		self.assertEqual(1, len(details))
		
		
	def test_layer_alpha(self):

		details = self.watcher.analyze(layers=[17])
		actual_alpha = details.alpha.to_numpy()[0]
		expected_alpha = 6.883741560463132
		self.assertAlmostEqual(actual_alpha,expected_alpha, delta=0.005)
		
		
	def test_xmax_set(self):
		"""test that xmax=-1 ignores the top eigenvalues"""
		details = self.watcher.analyze(layers=[17], xmax=-1)
		actual_alpha = details.alpha.to_numpy()[0]
		expected_alpha = 3.0
		self.assertAlmostEqual(actual_alpha,expected_alpha, delta=0.1 )
		
		
				
	def test_layer_alpha_w_powerlaw(self):

		details = self.watcher.analyze(layers=[17], pl_package=POWERLAW)
		actual_alpha = details.alpha.to_numpy()[0]
		expected_alpha = 6.883741560463132
		self.assertAlmostEqual(actual_alpha,expected_alpha, delta=0.005)
		
		
	def test_fix_fingers(self):

		details = self.watcher.analyze(layers=[17], fix_fingers='clip_xmax')
		actual_alpha = details.alpha.to_numpy()[0]
		expected_alpha = 3.0
		self.assertAlmostEqual(actual_alpha,expected_alpha, delta=0.1 )
		
		
	def test_fix_fingers_w_powerlaw(self):
			
		details = self.watcher.analyze(layers=[17], fix_fingers='clip_xmax', pl_package=POWERLAW_PACKAGE)
		actual_alpha = details.alpha.to_numpy()[0]
		expected_alpha = 3.0
		self.assertAlmostEqual(actual_alpha,expected_alpha, delta=0.1 )
		


#  https://kapeli.com/cheat_sheets/Python_unittest_Assertions.docset/Contents/Resources/Documents/index

class Test_VGG11_Base(Test_Base):
	"""
	layer_id    name     M  ...      longname  num_evals rf
0          2  Conv2d     3  ...    features.0         27  9
1          5  Conv2d    64  ...    features.3        576  9
2          8  Conv2d   128  ...    features.6       1152  9
3         10  Conv2d   256  ...    features.8       2304  9
4         13  Conv2d   256  ...   features.11       2304  9
5         15  Conv2d   512  ...   features.13       4608  9
6         18  Conv2d   512  ...   features.16       4608  9
7         20  Conv2d   512  ...   features.18       4608  9
8         25  Linear  4096  ...  classifier.0       4096  1
9         28  Linear  4096  ...  classifier.3       4096  1
10        31  Linear  1000  ...  classifier.6       1000  1
"""

	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_VGG11_Base:", self._testMethodName)
		
		self.params = DEFAULT_PARAMS.copy()
		# use older power lae
		self.params[PL_PACKAGE]=POWERLAW
		self.params[XMAX]=XMAX_FORCE


		self.model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)		
		
		self.first_layer = 2
		self.second_layer = 5
		self.third_layer = 8

		self.fc1_layer = 25
		self.fc2_layer = 28
		self.fc3_layer = 31
		
		self.fc_layers = [self.fc1_layer, self.fc2_layer, self.fc3_layer]
		self.min_layer_id = self.first_layer
		
	
		return

	def test_print_watcher(self):
		"""Issue #188"""
		
		results = self.watcher.results
		self.assertTrue(results is None)
		
		try:
			print(self.watcher)
		except:
			self.assertTrue(False, "watcher can not be printed")
			
					
		
	def test_layer_ids(self):
		"""Test that framework layer iteraor sets the layer ids as expected"""
		
		print(type(self.model))
		details = self.watcher.describe()
		print(details)
		
		layer_ids = details.layer_id.to_numpy()
		self.assertEqual(layer_ids[0], self.first_layer)
		self.assertEqual(layer_ids[1], self.second_layer)
		self.assertEqual(layer_ids[-3], self.fc1_layer)
		self.assertEqual(layer_ids[-2], self.fc2_layer)
		self.assertEqual(layer_ids[-1], self.fc3_layer)
		
		
		
	def test_get_framework_layer(self):
		"""Issue #165, get the underlying framework layer"""
		
		layer = self.watcher.get_framework_layer(layer=self.fc1_layer)
		print(type(layer))
		actual_layer_type = str(type(layer))
		expected_layer_type = "<class 'torch.nn.modules.linear.Linear'>"
		self.assertTrue(actual_layer_type, expected_layer_type)
		
			
	def test_basic_columns(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		
		details = self.watcher.describe()
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		for key in ['layer_id', 'name', 'M', 'N', 'Q', 'longname']:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))

		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		Q = details.Q.to_numpy()[0]

		self.assertAlmostEqual(Q, N/M, places=2)
		
		
		
	def test_basic_columns_with_model(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		
		details = self.watcher.describe(model=self.model)
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		for key in ['layer_id', 'name', 'M', 'N', 'Q']:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))

		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		Q = details.Q.to_numpy()[0]

		self.assertAlmostEqual(Q, N/M, places=2)


		
	def test_basic_columns_with_new_watcher(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		
		model =  models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		watcher = ww.WeightWatcher(log_level=logging.WARNING)
		
		details = watcher.describe(model=model)
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		for key in ['layer_id', 'name', 'M', 'N', 'Q']:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))

		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		Q = details.Q.to_numpy()[0]

		self.assertAlmostEqual(Q, N/M, places=2)


	def test_analyze_columns(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		

		details = self.watcher.analyze()
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		columns = "layer_id,name,D,M,N,alpha,alpha_weighted,has_esd,lambda_max,layer_type,log_alpha_norm,log_norm,log_spectral_norm,norm,num_evals,rank_loss,rf,sigma,spectral_norm,stable_rank,sv_max,xmax,xmin,num_pl_spikes,weak_rank_loss,fit_entropy".split(',')
		print(details.columns)
		for key in columns:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))
			
		
	def test_analyze_columns_with_model(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		

		details = self.watcher.analyze(model=self.model)
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		columns = "layer_id,name,D,M,N,alpha,alpha_weighted,has_esd,lambda_max,layer_type,log_alpha_norm,log_norm,log_spectral_norm,norm,num_evals,rank_loss,rf,sigma,spectral_norm,stable_rank,sv_max,xmax,xmin,num_pl_spikes,weak_rank_loss".split(',')
		print(details.columns)
		for key in columns:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))
			
		
	def test_mp_fit_columns(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		
		details = self.watcher.analyze(mp_fit=True)
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		columns = ["num_spikes", "sigma_mp", "mp_softrank"]
		
		for key in columns:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))
		
		
	#TODO: implement
	def test_random_mp_fit_columns_(self):
		"""N/A yet"""
		self.assertTrue(True)
		
	#TODO: implement
	def test_bad_params(self):
		"""N/A yet"""
		
		# only all positive or all negative layer id filters
		with self.assertRaises(Exception) as context:
			self.watcher.describe(layers=[-1,1])	
					
		#  deprecated: ww2x and kernel_fft 
		with self.assertRaises(Exception) as context:
			self.watcher.describe(kernel_fft=True)	
			
		#  deprecated: ww2x and kernel_fft , but not implemented as an error yet
		#with self.assertRaises(Exception) as context:
		#	self.watcher.describe(ww2x=True)	
						
		#  deprecated: ww2x and kernel_fft 
		with self.assertRaises(Exception) as context:
			self.watcher.describe(pool=False, kernel_fft=True)	

		# intra and kernel_fft
		with self.assertRaises(Exception) as context:
			self.watcher.describe(intra=True, kernel_fft=True)	
				
				
		# min_evals > max_evals
		with self.assertRaises(Exception) as context:
			self.watcher.describe(min_evals=100, max_evals=10)	

		# savefig is not a string or boolea
		with self.assertRaises(Exception) as context:
			self.watcher.describe(savefig=-1)	

		self.watcher.describe(savefig=True)
		self.watcher.describe(savefig='tmpdir')	
 
	def test_model_layer_types_ww2x(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers
		
			Note:  for 0.7 release, ww2x=True has been changed to pool=False
		"""
 
		details = self.watcher.describe(pool=False)
		
		denseLayers = details[details.layer_type==str(LAYER_TYPE.DENSE)]
		denseCount = len(denseLayers)
		self.assertEqual(denseCount, 3, "3 dense layers, but {} found".format(denseCount))
 			
	
		conv2DLayers = details[details.layer_type==str(LAYER_TYPE.CONV2D)]
		conv2DCount = len(conv2DLayers)
		self.assertEqual(conv2DCount, 8*9, "8*9 conv2D layers, but {} found".format(denseCount))
	
	def test_all_layer_types(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers"
		"""

		details = self.watcher.describe()
		
		denseLayers = details[details.layer_type==str(LAYER_TYPE.DENSE)]
		denseCount = len(denseLayers)
		self.assertEqual(denseCount, 3, "3 dense layers, but {} found".format(denseCount))		
	
		conv2DLayers = details[details.layer_type==str(LAYER_TYPE.CONV2D)]
		conv2DCount = len(conv2DLayers)
		self.assertEqual(conv2DCount, 8, "8 conv2D layers, but {} found".format(denseCount))
	
		
	def test_filter_dense_layer_types(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers"
		"""
		print("test_filter_dense_layer_types")
		details = self.watcher.describe(layers=[LAYER_TYPE.DENSE])
		print(details)
		
		denseLayers = details[details.layer_type==str(LAYER_TYPE.DENSE)]
		denseCount = len(denseLayers)

		self.assertEqual(denseCount, 3, "3 dense layers, but {} found".format(denseCount))
			
		# Dense layers are analyzed
		self.assertTrue((denseLayers.N > 0).all(axis=None), "All {} dense layers have a non zero N".format(denseCount))
		self.assertTrue((denseLayers.M > 0).all(axis=None), "All {} dense layers have a non zero M".format(denseCount))

		nonDenseLayers = details[details.layer_type!=str(LAYER_TYPE.DENSE)]
		nonDenseCount = len(nonDenseLayers)

		self.assertEqual(nonDenseCount, 0, "Filter has No dense layers: {} found".format(nonDenseCount))

		# Non Dense layers are NOT analyzed
		self.assertTrue((nonDenseLayers.N == 0).all(axis=None), "All {} NON dense layers have a zero N".format(nonDenseCount))
		self.assertTrue((nonDenseLayers.M == 0).all(axis=None), "All {} NON dense layers have a zero M".format(nonDenseCount))
		
			
	def test_filter_layer_ids(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers"
		"""
		
		details = self.watcher.describe(layers=[])
		print(details)
		
		details = self.watcher.describe(layers=self.fc_layers)
		print(details)
		
		denseLayers = details[details.layer_type==str(LAYER_TYPE.DENSE)]
		denseCount = len(denseLayers)
		self.assertEqual(denseCount, 3, "3 dense layers, but {} found".format(denseCount))
			
		nonDenseLayers = details[details.layer_type!=str(LAYER_TYPE.DENSE)]
		nonDenseCount = len(nonDenseLayers)
		self.assertEqual(nonDenseCount, 0, "Filter has No dense layers: {} found".format(nonDenseCount))

	def test_negative_filter_layer_ids(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers"
		"""
		
		skip_layers =  [-x for x in self.fc_layers]
		self.assertEqual(len(skip_layers), 3)
		print("skip these layers",skip_layers)

		details = self.watcher.describe(layers=[])
		print(details)
		
		details = self.watcher.describe(layers=skip_layers)
		print(details)
		
		denseLayers = details[details.layer_type==str(LAYER_TYPE.DENSE)]
		denseCount = len(denseLayers)
		self.assertEqual(denseCount, 0, " no dense layers, but {} found".format(denseCount))
			

	def test_filter_conv2D_layer_types(self):
		"""Test that ww.LAYER_TYPE.CONV2D filter is applied only to CONV2D layers"
		"""

		details = self.watcher.describe(layers=[ww.LAYER_TYPE.CONV2D])
		print(details)

		conv2DLayers = details[details['layer_type']==str(LAYER_TYPE.CONV2D)]
		conv2DCount = len(conv2DLayers)
		self.assertEqual(conv2DCount, 8, "# conv2D layers: {} found".format(conv2DCount))
		nonConv2DLayers = details[details['layer_type']!=str(LAYER_TYPE.CONV2D)]
		nonConv2DCount = len(nonConv2DLayers)
		self.assertEqual(nonConv2DCount, 0, "VGG11 has non conv2D layers: {} found".format(nonConv2DCount))

	

	def test_min_matrix_shape(self):
		"""Test that analyzes skips matrices smaller than  MIN matrix shape
		"""

		details = self.watcher.describe(min_evals=30)
		print(details)

		for nev in details.num_evals:
			self.assertGreaterEqual(nev, 30)
		

	def test_max_matrix_shape(self):
		"""Test that analyzes skips matrices larger than  MAX matrix shape
		"""

		details = self.watcher.describe(max_evals=1000)
		print(details)
		
		for nev in details.num_evals:
			self.assertLessEqual(nev, 1000)
		

	def test_describe_model(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		details = self.watcher.describe()
		print(details)
		self.assertEqual(len(details), 11)


 		
	def test_describe_model_ww2x(self):
		"""Test that alphas are computed and values are within thresholds
		
		Note:  ww2x=True has been replaced with pool=False
		"""
		details = self.watcher.describe(pool=False)
		self.assertEqual(len(details), 75)
		
		
	def test_dimensions(self):
		"""Test dimensions of Conv2D layer
		"""
		
			# default	
		details = self.watcher.describe()
		print(details)
		
		# default	
		details = self.watcher.describe(layers=[self.first_layer])
		print(details)
		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		rf = details.rf.to_numpy()[0]
		num_evals = details.num_evals.to_numpy()[0]
		print(N,M,rf,num_evals)
		
		self.assertEqual(N,64)
		self.assertEqual(M,3)
		self.assertEqual(rf,9)
		self.assertEqual(num_evals,M*rf)
		
	def test_switch_channels(self):
		"""Test user can switch the channels for a Conv2D layer
		"""
		# not available yet, experimental
		pass
	


	
	def test_compute_spectral_norms(self):
		"""Test that spectral norms computed and values are within thresholds
		
		ths may not work with the new FAST SVD...
		"""
		details = self.watcher.analyze(layers=[self.second_layer], pool=False, randomize=False, plot=False, mp_fit=False, svd_method=ACCURATE_SVD)

		# SLOW method
		a = details.spectral_norm.to_numpy()
		self.assertAlmostEqual(a[0],20.2149, places=4)
		self.assertAlmostEqual(a[1],24.8158, places=4)
		self.assertAlmostEqual(a[2],19.3795, places=4)
		
		
	def test_get_details(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		actual_details = self.watcher.analyze(layers=[self.second_layer])
		expected_details = self.watcher.get_details()
		
		self.assertEqual(len(actual_details), len(expected_details), "actual and expected details differ")
		
	def test_get_summary(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		details = self.watcher.analyze(layers=[self.second_layer])
		returned_summary = self.watcher.get_summary(details)
		
		print(returned_summary)
		
		saved_summary = self.watcher.get_summary()
		self.assertEqual(returned_summary, saved_summary)
		
	def test_get_summary_with_model(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		
		description = self.watcher.describe(model=self.model)
		self.assertEqual(11, len(description))
		
		
		details = self.watcher.analyze(model=self.model, layers=[self.second_layer])
		returned_summary = self.watcher.get_summary(details)
		
		print(returned_summary)
		
		saved_summary = self.watcher.get_summary()
		self.assertEqual(returned_summary, saved_summary)
		
		
	def test_get_summary_with_new_model(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		
		new_model =  models.vgg13(weights='VGG13_Weights.IMAGENET1K_V1')
		description = self.watcher.describe(model=new_model)
		self.assertEqual(13, len(description))
		
		fc3_layer = description.layer_id.to_numpy()[-1]
		details = self.watcher.analyze(model=new_model, layers=fc3_layer)
		returned_summary = self.watcher.get_summary(details)
		
		print(returned_summary)
		
		saved_summary = self.watcher.get_summary()
		self.assertEqual(returned_summary, saved_summary)


	def test_getESD(self):
		"""Test that eigenvalues are available in the watcher (no model specified here)
		"""

		print(self.watcher.describe())
		
		esd = self.watcher.get_ESD(layer=self.second_layer)
		self.assertEqual(len(esd), 576)

	def test_getESD_with_model(self):
		"""Test that eigenvalues are available while specifying the model explicitly
		"""

		esd = self.watcher.get_ESD(model=self.model, layer=self.second_layer)
		self.assertEqual(len(esd), 576)

	def test_randomize(self):
		"""Test randomize option : only checks that the right columns are present, not the values
		"""
		
		rand_columns = ['max_rand_eval', 'rand_W_scale', 'rand_bulk_max',
					 'rand_bulk_min', 'rand_distance', 'rand_mp_softrank', 
					 'rand_num_spikes', 'rand_sigma_mp']
       
		details = self.watcher.analyze(layers = [self.fc2_layer], randomize=False)	
		for column in rand_columns:
			self.assertNotIn(column, details.columns)
			
		details = self.watcher.analyze(layers = [self.fc2_layer], randomize=True)	
		for column in rand_columns:	
			self.assertIn(column, details.columns)


				
	def test_rand_distance(self):
		"""
		Test rand distance Not very accuracte since it is random
		"""
		
		details= self.watcher.analyze(layers=[self.fc2_layer], randomize=True)
		actual = details.rand_distance[0]
		expected = 0.29
		self.assertAlmostEqual(actual,expected, places=2)

	def test_ww_softrank(self):
		"""
		   Not very accuracte since it relies on randomizing W
		"""
		#
		details= self.watcher.analyze(layers=[self.fc3_layer], randomize=True, mp_fit=True)
		print(details[['ww_softrank','mp_softrank', 'lambda_max', 'rand_bulk_max', 'max_rand_eval']])
		actual = details.ww_softrank[0]
		expected = details.mp_softrank[0]
		self.assertAlmostEqual(actual,expected, places=1)
		
		max_rand_eval = details.max_rand_eval[0]
		max_eval = details.lambda_max[0]
		expected = max_rand_eval/max_eval
		self.assertAlmostEqual(actual,expected, places=2)

	
	def test_ww_maxdist(self):
		"""
		   Not very accuracte since it relies on randomizing W
		"""
		
		details= self.watcher.analyze(layers=[self.fc2_layer], randomize=True)
		print(details)
		actual = details.ww_maxdist[0]/100.0
		expected = 39.9/100.0
		self.assertAlmostEqual(actual,expected, places=2)
		
	def test_reset_params(self):
		"""test that params are reset / normalized ()
	
		note: for 0.7, this test was modified to include the PL_PACKAGE param"""
		
		params = DEFAULT_PARAMS.copy()
		params['fit']=PL
		valid = self.watcher.valid_params(params)
		self.assertTrue(valid)
		params = self.watcher.normalize_params(params)
		self.assertEqual(params['fit'], POWER_LAW)
		
		params = DEFAULT_PARAMS.copy()
		params['fit']=TPL
		valid = self.watcher.valid_params(params)
		self.assertFalse(valid)
		
		params = DEFAULT_PARAMS.copy()
		params['fit']=TPL
		params[PL_PACKAGE]=POWERLAW_PACKAGE
		params[XMAX]=XMAX_FORCE

		valid = self.watcher.valid_params(params)
		self.assertTrue(valid)
		params = self.watcher.normalize_params(params)
		self.assertEqual(params['fit'], TRUNCATED_POWER_LAW)
		
		


	
	def test_density_fit(self):
		"""Test the fitted sigma from the density fit: FIX
		
		There is something wrong with these fits, the sigma is way too low
		"""
 		
		details = self.watcher.analyze(layers = [self.fc1_layer], pool=False, randomize=False, plot=False, mp_fit=True)
		num_spikes = details.num_spikes.to_numpy()[0]
		sigma_mp = details.sigma_mp.to_numpy()[0]
		#mp_softrank = details.mp_softrank.to_numpy()[0])
		
		print("num spikes", num_spikes)
		print("sigma mp", sigma_mp)
		
		print("I think something is wrong here")

		#self.assertAlmostEqual(num_spikes, 672) #numofSig
		#self.assertAlmostEqual(sigma_mp, 0.7216) #sigma_mp
		#self.assertAlmostEqual(details.np_softrank, 0.203082, places = 6) 
		
	def test_density_fit_on_randomized_W(self):
		"""Test the fitted sigma from the density fit: FIX
		
		"""
 		
		details = self.watcher.analyze(layers = [self.fc1_layer], pool=False, randomize=True, plot=False, mp_fit=True)
		rand_num_spikes = details.rand_num_spikes.to_numpy()[0]
		rand_sigma_mp = details.rand_sigma_mp.to_numpy()[0]
		#mp_softrank = details.mp_softrank.to_numpy()[0])
		
		print("rand_num_spikes", rand_num_spikes)  # correlation trap ?
		print("rand_sigma mp", rand_sigma_mp)

		self.assertAlmostEqual(rand_num_spikes, 1) #numofSig
		self.assertAlmostEqual(rand_sigma_mp, 1.0, delta=0.0001) #sigma_mp
		#self.assertAlmostEqual(details.np_softrank, 0.203082, places = 6) 


	def test_svd_smoothing(self):
		"""Test the svd smoothing on layer FC2 of VGG"""
		
		# 819 =~ 4096*0.2
		self.watcher.SVDSmoothing(layers=[self.fc2_layer])
		esd = self.watcher.get_ESD(layer=self.fc2_layer) 
		num_comps = len(esd[esd > 10**-10])
		self.assertEqual(num_comps, 819)
		
		
	def test_svd_smoothing_with_model(self):
		"""Test the svd smoothing on layer FC2 of VGG"""

		
		# 819 =~ 4096*0.2
		self.watcher.SVDSmoothing(model=self.model, layers=[self.fc2_layer])
		esd = self.watcher.get_ESD(layer=self.fc2_layer) 
		num_comps = len(esd[esd>10**-10])
		self.assertEqual(num_comps, 819)


	def test_svd_smoothing_alt(self):
		"""Test the svd smoothing on 1 layer of VGG
		The intent is that test_svd_smoothing and test_svd_smoothing_lat are exactly the same
		except that:

		test_svd_smoothing() only applies TruncatedSVD, and can only keep the top N eigenvectors

		whereas

		test_svd_smoothing_alt() allows for a negative input, which throws away the top N eigenvectors

		"""
 		
		print("----test_svd_smoothing_alt-----")

		# need model here; somehow self.model it gets corrupted by SVD smoothing
		#model = models.vgg11(pretrained=True)
		
		self.watcher.SVDSmoothing(layers=[self.fc2_layer], percent=-0.2)
		esd = self.watcher.get_ESD(layer=self.fc2_layer) 
		num_comps = len(esd[esd>10**-10])
		# 3277 = 4096 - 819
		print("num comps = {}".format(num_comps))
		self.assertEqual(num_comps, 3277)
		
	def test_svd_smoothing_alt2(self):
		"""Test the svd smoothing on 1 layer of VGG
		
		"""
 		
		print("----test_svd_smoothing_alt2-----")
		
		# need model here; somehow self.model it gets corrupted by SVD smoothing
		#model = models.vgg11(pretrained=True)
		
		self.watcher.SVDSmoothing(layers=[self.fc2_layer], percent=0.2)
		esd = self.watcher.get_ESD(layer=self.fc2_layer) 
		num_comps = len(esd[esd>10**-10])
		# 3277 = 4096 - 819
		print("num comps = {}".format(num_comps))
		self.assertEqual(num_comps, 819)
		
		
		
	def test_svd_sharpness(self):
		"""Test the svd smoothing on 1 layer of VGG
		"""
 			
		esd_before = self.watcher.get_ESD(layer=self.fc2_layer) 
		
		self.watcher.SVDSharpness(layers=[self.fc2_layer])
		esd_after = self.watcher.get_ESD(layer=self.fc2_layer) 
		
		print("max esd before {}".format(np.max(esd_before)))
		print("max esd after {}".format(np.max(esd_after)))

		self.assertGreater(np.max(esd_before)-2.0,np.max(esd_after))
		
		
			
		
	def test_svd_sharpness_with_model(self):
		"""Test the svd smoothing on 1 layer of VGG
		"""
 		
		esd_before = self.watcher.get_ESD(model=self.model, layer=self.fc2_layer) 
		
		self.watcher.SVDSharpness(layers=[self.fc2_layer])
		esd_after = self.watcher.get_ESD(layer=self.fc2_layer) 
		
		print("max esd before {}".format(np.max(esd_before)))
		print("max esd after {}".format(np.max(esd_after)))

		self.assertGreater(np.max(esd_before),np.max(esd_after))
		
		
				
	def test_ESD_model_set(self):
		"""
		Test that we can get the ESD when setting the model explicitly
		""" 
	
		details = self.watcher.describe(model=self.model)
		print(details)
		 
		esd_before = self.watcher.get_ESD(model=self.model, layer=self.fc2_layer) 
		esd_after = self.watcher.get_ESD(layer=self.fc2_layer) 
		
		print(len(esd_before))
		print(len(esd_after))

		
		print("max esd before {}".format(np.max(esd_before)))
		print("max esd after {}".format(np.max(esd_after)))

		self.assertEqual(np.max(esd_before),np.max(esd_after))
		self.assertEqual(np.min(esd_before),np.min(esd_after))

		
	def test_svd_sharpness2(self):
		"""Test the svd smoothing on 1 layer of VGG
		"""
 		
		print("----test_svd_sharpness-----")

		#model = models.vgg11(pretrained=True)
		model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')

		self.watcher = ww.WeightWatcher(model=model, log_level=logging.WARNING)
		
		
		esd_before = self.watcher.get_ESD(layer=self.third_layer) 
		
		self.watcher.SVDSharpness(layers=[self.third_layer])
		esd_after = self.watcher.get_ESD(layer=self.third_layer) 
		
		print("max esd before {}".format(np.max(esd_before)))
		print("max esd after {}".format(np.max(esd_after)))

		self.assertGreater(np.max(esd_before),np.max(esd_after))
		
		

	def test_runtime_warnings(self):
		"""Test that runtime warnings are still active
		"""
		print("test runtime warning: sqrt(-1)=", np.sqrt(-1.0))
		assert(True)
		
	def test_N_ge_M(self):
		"""Test that the Keras on VGG11 M,N set properly on Conv2D layers
		"""
		details = self.watcher.describe()
		M = details.M.to_numpy()
		N = details.N.to_numpy()
		self.assertTrue((N >= M).all)
	
	def test_num_evals(self):
		"""Test that the num evals is correct
		"""
		details = self.watcher.describe()		
		self.assertTrue((details.M * details.rf == details.num_evals).all())
		
	def test_rf_value(self):
		"""Test that the num evals is correct
		"""
		details = self.watcher.describe()		
		self.assertTrue((details.rf.to_numpy()[0:8] == 9.0).all())
		
		
	def test_randomize_mp_fits(self):
		"""Test that the mp_fits works correctly for the randomized matrices
		Note: the fits currently not consistent
		"""
		details = self.watcher.analyze(mp_fit=True,  randomize=True,  pool=True)
		self.assertTrue((details.rand_sigma_mp < 1.10).all())
		self.assertTrue((details.rand_sigma_mp > 0.96).all())
		self.assertTrue((details.rand_num_spikes.to_numpy() < 80).all())
		
		
	
	def test_make_ww_iterator(self):
		"""Test that we can make the default ww layer iterator
		"""
		details = self.watcher.describe()
		actual_num_layers = len(details)
		expected_num_layers = 11
		expected_ids = details.layer_id.to_numpy().tolist()

		self.assertEqual(actual_num_layers, expected_num_layers)
		self.assertEqual(len(expected_ids), expected_num_layers)
		
		params = DEFAULT_PARAMS.copy()
		params[MIN_EVALS] = 0
		
		iterator = self.watcher.make_layer_iterator(model=self.model, params=params)
		num = 0
		actual_ids = []
		for ww_layer in iterator:
			self.assertGreater(ww_layer.layer_id,self.min_layer_id-1)
			actual_ids.append(ww_layer.layer_id)
			num += 1
		self.assertEqual(num,11)
		self.assertEqual(actual_ids,expected_ids)

		
		iterator = self.watcher.make_layer_iterator(model=self.model, layers=[self.fc2_layer])
		num = 0
		for ww_layer in iterator:
			self.assertEqual(ww_layer.layer_id,self.fc2_layer)
			num += 1
		self.assertEqual(num,1)


	
	def test_start_ids_1(self):
		"""same as  test_make_ww_iterator, but checks that the ids can start at 1, not 0
		
		"""
		
		details = self.watcher.describe()
		print(details, len(details))
		actual_num_layers = len(details)
		expected_num_layers = 11
		expected_ids = details.layer_id.to_numpy().tolist()
		expected_ids = [x+1 for x in expected_ids]

		self.assertEqual(actual_num_layers, expected_num_layers)
		self.assertEqual(len(expected_ids), expected_num_layers)


		# test decribe
		details = self.watcher.describe(start_ids=1)
		print(details)
		actual_ids = details.layer_id.to_numpy().tolist()
		self.assertEqual(actual_ids,expected_ids)

		# test analyze: very slow
		# details = self.watcher.analyze(start_ids=1)
		# actual_ids = details.layer_id.to_numpy().tolist()
		# self.assertEqual(actual_ids,expected_ids)

		params = DEFAULT_PARAMS.copy()
		params[START_IDS]=1
		params[MIN_EVALS]=1 # there may be a side effect that resets this
		
		# test iterator
		iterator = self.watcher.make_layer_iterator(model=self.model, params=params)
		num = 0
		actual_ids = []
		for ww_layer in iterator:
			self.assertGreater(ww_layer.layer_id,0)
			actual_ids.append(ww_layer.layer_id)
			num += 1
			print(num, ww_layer.layer_id)
		self.assertEqual(num,11)
		self.assertEqual(actual_ids,expected_ids)

	
		
	# CHM:  stacked layers may not be working properly, be careful
	# needs more testing 
	def test_ww_stacked_layer_iterator(self):
		"""Test Stacked Layer Iterator
		"""
				
		params = DEFAULT_PARAMS.copy()
		params['stacked'] = True
		iterator = self.watcher.make_layer_iterator(model=self.model, params=params)
		#TODO: get this to work!
		#self.assertEqual(iterator.__class__.__name__, WWStackedLayerIterator)
		num = 0
		for ww_layer in iterator:
			num+=1
			
		self.assertEqual(num,1)
		self.assertEqual(ww_layer.name, "Stacked Layer")
		self.assertEqual(ww_layer.layer_id,0)
	#	self.assertEqual(ww_layer.N,29379) ?
	#	self.assertEqual(ww_layer.M,25088) ?
		self.assertEqual(ww_layer.rf,1)
	#	self.assertEqual(ww_layer.num_components,ww_layer.M)
		
		
	def test_ww_stacked_layer_details(self):
		"""Test Stacked Layer Iterator
		"""
				
		details = self.watcher.describe(model=self.model, stacked=True)
		self.assertEqual(len(details),1)
	#	self.assertEqual(details.N.to_numpy()[0],29379)
	#	self.assertEqual(details.M.to_numpy()[0],25088)
		self.assertEqual(details.rf.to_numpy()[0],1)
		self.assertEqual(details.layer_type.to_numpy()[0],str(LAYER_TYPE.STACKED))
		
		
				
	def test_rescale_eigenvalues(self):	
		"""test rescaling un rescaling evals"""
		
		evals = self.watcher.get_ESD(layer=self.fc2_layer)
		rescaled_evals, Wscale = RMT_Util.rescale_eigenvalues(evals)
		un_rescaled_evals = RMT_Util.un_rescale_eigenvalues(rescaled_evals, Wscale)

		actual = np.max(evals)
		expected =  np.max(un_rescaled_evals)
		self.assertAlmostEqual(actual, expected)

		
		
	def test_permute_W(self):
		"""Test that permute and unpermute methods work
		"""
		N, M = 4096, 4096
		iterator = self.watcher.make_layer_iterator(layers=[self.fc2_layer])
		for ww_layer in iterator:
			self.assertEqual(ww_layer.layer_id,self.fc2_layer)
			W = ww_layer.Wmats[0]
			self.assertEqual(W.shape,(N,M))
			
			self.watcher.apply_permute_W(ww_layer)
			W2 = ww_layer.Wmats[0]
			self.assertNotEqual(W[0,0],W2[0,0])
			
			self.watcher.apply_unpermute_W(ww_layer)
			W2 = ww_layer.Wmats[0]
			self.assertEqual(W2.shape,(N,M))
			self.assertEqual(W[0,0],W2[0,0])
			
			
				
	def test_permute_W_with_model(self):
		"""Test that permute and unpermute methods work
		"""
		N, M = 4096, 4096
		iterator = self.watcher.make_layer_iterator(model=self.model, layers=[self.fc2_layer])
		for ww_layer in iterator:
			self.assertEqual(ww_layer.layer_id,self.fc2_layer)
			W = ww_layer.Wmats[0]
			self.assertEqual(W.shape,(N,M))
			
			self.watcher.apply_permute_W(ww_layer)
			W2 = ww_layer.Wmats[0]
			self.assertNotEqual(W[0,0],W2[0,0])
			
			self.watcher.apply_unpermute_W(ww_layer)
			W2 = ww_layer.Wmats[0]
			self.assertEqual(W2.shape,(N,M))
			self.assertEqual(W[0,0],W2[0,0])
			
	#TODO: add test fit entropy
	def test_fit_entropy_on_layer(self):
		details = self.watcher.analyze(model=self.model, layers=[self.fc2_layer])
		expected = details.fit_entropy.to_numpy()[0]
		
		import powerlaw
		esd = self.watcher.get_ESD(layer=self.fc2_layer)
		fit = powerlaw.Fit(esd,  xmax=None)
		actual = RMT_Util.line_entropy(fit.Ds)
		self.assertAlmostEqual(expected, actual, None, '', 0.01)

		
	def test_same_models(self):
		""" test same models (not finished yet) """
		
		# TODO: finish
		pass




class Test_VGG11_StateDict(Test_VGG11_Base):
	"""All the same tests as for VGG11, but using the statedict option"""
	
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_VGG11_StateDict:", self._testMethodName)
		
		self.params = DEFAULT_PARAMS.copy()
		# use older power lae
		self.params[PL_PACKAGE]=POWERLAW
		self.params[XMAX]=XMAX_FORCE
		
		self.model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1').state_dict()
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)

		self.first_layer = 1
		self.second_layer = 2
		self.third_layer = 8
		self.fc1_layer = 9
		self.fc2_layer = 10
		self.fc3_layer = 11
		
		self.fc_layers = [self.fc1_layer, self.fc2_layer, self.fc3_layer]
		self.min_layer_id = self.first_layer
		
		return

	def test_start_ids_1(self):
		"""same as  test_make_ww_iterator, but checks that the ids can start at 1, not 0
		   BUT FOR PyStateDict, we ALWAYS start at 1
		"""
		
		details = self.watcher.describe()
		print(details, len(details))
		actual_num_layers = len(details)
		expected_num_layers = 11
		expected_ids = details.layer_id.to_numpy().tolist()

		self.assertEqual(actual_num_layers, expected_num_layers)
		self.assertEqual(len(expected_ids), expected_num_layers)


		# test decribe
		details = self.watcher.describe(start_ids=1)
		print(details)
		actual_ids = details.layer_id.to_numpy().tolist()
		self.assertEqual(actual_ids,expected_ids)
			

	
		
		
		
class Test_VGG11_Alpha_w_PowerLawFit(Test_Base):	
	"""Tests the alpha calculations on VGG11 (pytorch) using the old powerlaw package, xmax='force'"""
	
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_VGG11_Alpha_w_PowerLawFit:", self._testMethodName)
		
		self.params = DEFAULT_PARAMS.copy()
		# use older power lae
		self.params[PL_PACKAGE]=POWERLAW
		self.params[XMAX]=XMAX_FORCE


		self.model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)
		
		self.first_layer = 2
		self.second_layer = 5
		self.third_layer = 8

		self.fc1_layer = 25
		self.fc2_layer = 28
		self.fc3_layer = 31
		
		self.fc_layers = [self.fc1_layer, self.fc2_layer, self.fc3_layer]
		self.min_layer_id = self.first_layer
		
	
		return
	
	def test_powerlaw_package_available(self):
		"""Test that the powerlaw package is available"""
		import importlib

		try:
			importlib.import_module('powerlaw')
		except ImportError:
			self.fail("Failed to import powerlaw package")


	
	## TODO:
	#  add layers, pool=False/False
	
	def test_compute_alphas(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		details = self.watcher.analyze(layers=[self.second_layer], pool=False, randomize=False, plot=False, mp_fit=False, 
									svd_method=ACCURATE_SVD, pl_package=POWERLAW_PACKAGE, xmax=XMAX_FORCE)
		#d = self.watcher.get_details(results=results)_method
		a = details.alpha.to_numpy()
		self.assertAlmostEqual(a[0],1.65014, places=4)
		self.assertAlmostEqual(a[1],1.57297, places=4)
		self.assertAlmostEqual(a[3],1.43459, places=4)
		
		# WHY DPOES THIS TEST FAIL NOW ?


	
		details2 = self.watcher.analyze(layers=[self.second_layer], pool=False, randomize=False, plot=False, mp_fit=False,  
									pl_package=POWERLAW_PACKAGE, xmax=None)
		#d = self.watcher.get_details(results=results)WW_
		a2 = details2.alpha.to_numpy()
		self.assertAlmostEqual(a2[0],1.74859, places=4)
		self.assertAlmostEqual(a2[1],1.66595, places=4)
		self.assertAlmostEqual(a2[3],1.43459, places=4)
 	
		
	def test_intra_power_law_fit(self):
		"""Test PL fits on intra
		"""

		print(type(self.fc_layers[0:2]), self.fc_layers[0:2])
		details= self.watcher.analyze(layers=self.fc_layers[0:2], intra=True, randomize=False, vectors=False, pl_package=POWERLAW_PACKAGE, xmax=XMAX_FORCE)
		actual_alpha = details.alpha[0]
		actual_best_fit = details.best_fit[0]
		print(actual_alpha,actual_best_fit)

		expected_alpha =  2.654 # not very accurate because of the sparisify transform
		expected_best_fit = LOG_NORMAL
		self.assertAlmostEqual(actual_alpha,expected_alpha, places=1)
		self.assertEqual(actual_best_fit, expected_best_fit)
		
		
	def test_intra_power_law_fit2(self):
		"""Test PL fits on intram, sparsify off, more accurate
			"""
		print(type(self.fc_layers[0:2]), self.fc_layers[0:2])
		details= self.watcher.analyze(layers=self.fc_layers[0:2], intra=True, sparsify=False, pl_package=POWERLAW_PACKAGE, xmax=XMAX_FORCE)
		actual_alpha = details.alpha[0]
		actual_best_fit = details.best_fit[0]
		print(actual_alpha,actual_best_fit)


		expected_alpha =  2.719 # close to exact ?
		expected_best_fit = LOG_NORMAL
		self.assertAlmostEqual(actual_alpha,expected_alpha, places=2)
		self.assertEqual(actual_best_fit, expected_best_fit)

	def test_truncated_power_law_fit(self):
		"""Test TPL fits:  note that the new toprch method reduces the accureacy of the test
		"""
		
		# need model here; somehow self.model it gets corrupted by SVD smoothing
		#model = models.vgg11(pretrained=True)
		model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')

		self.watcher = ww.WeightWatcher(model=model, log_level=logging.WARNING)
		
		details= self.watcher.analyze(layers=[self.fc2_layer], fit='TPL', pl_package=POWERLAW_PACKAGE, xmax=XMAX_FORCE)
		actual_alpha = details.alpha[0]
		actual_Lambda = details.Lambda[0]

		self.assertTrue(actual_Lambda > -1) #Lambda must be set for TPL

		# these numbers have not been independently verified yet
		expected_alpha = 2.1
		delta = 0.1
		self.assertAlmostEqual(actual_alpha,expected_alpha, None, '',  delta)
		expected_Lambda =  0.017
		delta = 0.001
		self.assertAlmostEqual(actual_Lambda,expected_Lambda, None, '',  delta)
		
		
	def test_extended_truncated_power_law_fit(self):
		"""Test E-TPL fits.  Runs TPL with fix_fingets = XMIN_PEAK
		"""
		details= self.watcher.analyze(layers=[self.fc1_layer], fit=E_TPL, pl_package=POWERLAW_PACKAGE, xmax=XMAX_FORCE)
		actual_alpha = details.alpha[0]
		actual_Lambda = details.Lambda[0]

		self.assertTrue(actual_Lambda > -1) #Lambda must be set for TPL
		
		# these numbers have not been independently verified yet
		expected_alpha = 2.07
		expected_Lambda =  0.02
		self.assertAlmostEqual(actual_alpha,expected_alpha, places=2)
		self.assertAlmostEqual(actual_Lambda,expected_Lambda, places=2)
		 
		
		
	def test_fix_fingers_xmin_peak(self):
		"""Test fix fingers xmin_peak.  Again, notice that wiothj and without FORCE give slightly different results
		"""
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)
			
		# default
		details = self.watcher.analyze(layers=[self.second_layer], xmax=FORCE, pl_package=POWERLAW_PACKAGE)
		actual = details.alpha.to_numpy()[0]
		expected = 7.116304
		print("ACTUAL {}".format(actual))
		self.assertAlmostEqual(actual,expected, places=2)

		# XMIN_PEAK xmax FORCED
		details = self.watcher.analyze(layers=[self.second_layer], fix_fingers='xmin_peak', xmax=FORCE, xmin_max=1.0, pl_package=POWERLAW_PACKAGE)
		actual = details.alpha[0]
		actual = details.alpha.to_numpy()[0]
		expected = 1.68
		delta = 0.01
		self.assertAlmostEqual(actual,expected, None, '',  delta)
		
		
		# XMIN_PEAK xmax None, sligltly different alphja
		details = self.watcher.analyze(layers=[self.second_layer], fix_fingers='xmin_peak', xmin_max=1.0, pl_package=POWERLAW_PACKAGE)
		actual = details.alpha[0]
		actual = details.alpha.to_numpy()[0]
		expected = 1.72
		delta = 0.01
		self.assertAlmostEqual(actual,expected, None, '',  delta)
	
		
	def test_fix_fingers_clip_xmax_w_Force(self):
		"""Test fix fingers clip_xmax:  if we force xmax=np.max(evals), we get a finger here
		"""
		
		details = self.watcher.analyze(layers=[self.second_layer], xmax=FORCE, pl_package=POWERLAW_PACKAGE)
		actual = details.alpha.to_numpy()[0]
		expected = 7.12
		self.assertAlmostEqual(actual,expected, places=2)
		
		# CLIP_XMAX FORCED
		details = self.watcher.analyze(layers=[self.second_layer], xmax=FORCE, fix_fingers='clip_xmax', pl_package=POWERLAW_PACKAGE)
		actual = details.alpha.to_numpy()[0]
		expected = 1.67
		self.assertAlmostEqual(actual,expected, places=2)
		
		num_fingers = details.num_fingers.to_numpy()[0]
		self.assertEqual(num_fingers,1)



	def test_fix_fingers_clip_xmax_None(self):
		"""Test fix fingers clip_xmax:  Note that there are NO fingers with xmax=None 
		"""
		
		
		details = self.watcher.analyze(layers=[self.second_layer], pl_package=POWERLAW_PACKAGE)
		actual = details.alpha.to_numpy()[0]
		expected = 1.72
		self.assertAlmostEqual(actual,expected, places=2)
		
		# CLIP_XMAX,=None is default, and there is no finger
		details = self.watcher.analyze(layers=[self.second_layer], fix_fingers='clip_xmax', pl_package=POWERLAW_PACKAGE)
		actual = details.alpha.to_numpy()[0]
		expected = 1.72
		self.assertAlmostEqual(actual,expected, places=2)
		
		num_fingers = details.num_fingers.to_numpy()[0]
		self.assertEqual(num_fingers,0)
		
		# CLIP_XMAX, =None explicit
		details = self.watcher.analyze(layers=[self.second_layer], fix_fingers='clip_xmax', pl_package=POWERLAW_PACKAGE, xmax=None)
		actual = details.alpha.to_numpy()[0]
		expected = 1.72
		self.assertAlmostEqual(actual,expected, places=2)
		
		num_fingers = details.num_fingers.to_numpy()[0]
		self.assertEqual(num_fingers,0)

	
	
class Test_VGG11_Alpha_w_WWFit(Test_Base):	
	"""Tests the  alpha calculations on VGG11 (pytorch) using the WWFit class'"""
	
		
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_VGG11_Alpha_w_WWFit:", self._testMethodName)
		
		self.params = DEFAULT_PARAMS.copy()

		self.model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)
		
		self.first_layer = 2
		self.second_layer = 5
		self.third_layer = 8

		self.fc1_layer = 25
		self.fc2_layer = 28
		self.fc3_layer = 31
		
		self.fc_layers = [self.fc1_layer, self.fc2_layer, self.fc3_layer]
		self.min_layer_id = self.first_layer
		
		
	def test_WWFit_available(self):
		try:
			from weightwatcher.WW_powerlaw import WWFit
		except ImportError:
	 		self.fail("Failed to import WWFit class from WW_powerlaw module")
	 		
	 		
	def test_compute_alphas(self):
		"""Test that alphas are computed and values are within thresholds using the XMax=None option
		"""
		details = self.watcher.analyze(layers=[self.second_layer], pool=False, randomize=False, plot=False, mp_fit=False, pl_package=WW_POWERLAW_PACKAGE)
		#d = self.watcher.get_details(results=results)
		a = details.alpha.to_numpy()
		self.assertAlmostEqual(a[0],1.74859, places=4)
		self.assertAlmostEqual(a[1],1.66595, places=4)
		self.assertAlmostEqual(a[3],1.43459, places=4)
		
		
	#
	# TODO: check if xmax='force' does anything ?
	#
	def test_fix_fingers_clip_xmax_w_Force(self):
		"""Test fix fingers clip_xmax
		
				this should give the same answer as the old POWERLAW_PACKAGE, with xmax=FORCE

		"""
		
		# CLIP_XMAX
		details = self.watcher.analyze(layers=[self.second_layer], xmax='force', fix_fingers='clip_xmax', pl_package=POWERLAW_PACKAGE)
		actual = details.alpha.to_numpy()[0]
		expected = 1.6635
		self.assertAlmostEqual(actual,expected, places=2)
		
		num_fingers = details.num_fingers.to_numpy()[0]
		self.assertEqual(num_fingers,1)
		
		
	def test_fix_fingers_clip_xmax(self):
		"""Test fix fingers clip_xmax
		
		this should give the same answer as the old POWERLAW_PACKAGE, with xmax=None and has NO fingers
		"""
		
		# CLIP_XMAX
		details = self.watcher.analyze(layers=[self.second_layer],  fix_fingers='clip_xmax', pl_package=WW_POWERLAW_PACKAGE)
		actual = details.alpha.to_numpy()[0]
		expected = 1.72
		self.assertAlmostEqual(actual,expected, places=2)
		
		num_fingers = details.num_fingers.to_numpy()[0]
		self.assertEqual(num_fingers,0)
		
	def test_kernel_fft(self):
		"""Test the FFT method"; why does this fail ? """
		
		details = self.watcher.describe(layers=[self.first_layer], kernel_fft=True)
		print(details)
		
		details = self.watcher.analyze(layers=[self.first_layer], kernel_fft=True)
		actual = details.alpha.to_numpy()[0]
		expected = 2.144
		self.assertAlmostEqual(actual,expected, places=4)
		
		
class Test_VGG11_StateDict_Alpha_w_WWFit(Test_VGG11_Alpha_w_WWFit):	
	"""Tests the  alpha calculations on VGG11 (pytorch, statedict format) using the WWFit class'"""	
	
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_VGG11_StateDict_Alpha_w_WWFit:", self._testMethodName)
		
		self.params = DEFAULT_PARAMS.copy()
		# use older power lae
		self.params[PL_PACKAGE]=POWERLAW
		self.params[XMAX]=XMAX_FORCE
		
		self.model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1').state_dict()
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)

		self.first_layer = 1
		self.second_layer = 2
		self.third_layer = 8
		self.fc1_layer = 9
		self.fc2_layer = 10
		self.fc3_layer = 11
		
		self.fc_layers = [self.fc1_layer, self.fc2_layer, self.fc3_layer]
		self.min_layer_id = self.first_layer
		
		return
	
	

class Test_Keras(Test_Base):
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_Keras:", self._testMethodName)
		self.model = VGG16()
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)
		
	def test_kayer_ids(self):
		details = self.watcher.describe()
		print(details)
		

	def test_basic_columns(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		
		details = self.watcher.describe()
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		for key in ['layer_id', 'name', 'M', 'N', 'Q', 'longname']:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))

		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		Q = details.Q.to_numpy()[0]

		self.assertAlmostEqual(Q, N/M, places=2)

	def test_num_layers(self):
		"""Test that the Keras on VGG11
		"""
		details = self.watcher.describe()
		print("Testing Keras on VGG16")
		self.assertEqual(len(details), 16)


	def test_num_layers_with_model(self):
		"""Test that the Keras on VGG11
		"""
		details = self.watcher.describe(model=self.model)
		print("Testing Keras on VGG16")
		self.assertEqual(len(details), 16)
		

	def test_channels_first(self):
		"""Test that the Keras on VGG11 M,N set properly on Conv2D layers
		"""
		details = self.watcher.describe()
		M = details.iloc[0].M
		N = details.iloc[0].N
		self.assertEqual(M, 3)
		self.assertEqual(N, 64)
		
	def test_N_ge_M(self):
		"""Test that the Keras on VGG11 M,N set properly on Conv2D layers
		"""
		details = self.watcher.describe()
		M = details.M.to_numpy()
		N = details.N.to_numpy()
		self.assertTrue((N >= M).all)


	def test_layer_ids(self):
		"""Test that the layer_ids are signed as expected
		
		im not sure we are doing this consistantly with how we did it begore"""
		
		details = self.watcher.describe()
		print(details)
		actual_layer_ids = list(details.layer_id.to_numpy())
		expected_layer_ids = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 20, 21, 22]
		self.assertListEqual(actual_layer_ids, expected_layer_ids)
		
	def test_keras_model_with_no_bias(self):
		"""Resolve uissue #201 to treat
		
		keras neural network contains dense layers WITHOUT BIAS TERM
		
		"""
		
		from keras.models import Sequential
		from keras.layers import Dense, Activation
		
		model = Sequential()
		model.add(Dense(100, input_shape=(190,), use_bias=False))
		model.add(Activation("relu"))
		model.add(Dense(10, use_bias=False))
		model.add(Activation('sigmoid'))
		
		watcher = ww.WeightWatcher(model=model)
		details = watcher.describe()
		print(details)
		self.assertTrue(len(details)==2)
		
		details = watcher.analyze()
		print(details)
		self.assertTrue(len(details)==1)
		
		
		
        
class Test_ResNet(Test_Base):
	def setUp(self):
		"""I run before every test in this class
		"""
		self.model = models.resnet18()
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)

		
	def test_N_ge_M(self):
		"""Test that the Keras on VGG11 M,N set properly on Conv2D layers
		"""
		details = self.watcher.describe()
		M = details.M.to_numpy()
		N = details.N.to_numpy()
		self.assertTrue((N >= M).all)
		
		
	def test_num_evals(self):
		"""Test that the num evals is correct
		"""
		details = self.watcher.describe()		
		self.assertTrue((details.M * details.rf == details.num_evals).all())
		



class Test_ResNet_Models(Test_Base):
	"""Replaces old Test_ResNet with SubTests and multple models"""
	
    
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_ResNet_Models:", self._testMethodName)
			
		self.models = [models.resnet18(), models.resnet18().state_dict()]
		self.model_names = ["resnet18()", "resnet18().state_dict()"]
		
		
	def test_weight_watcher(self):
		for i, model in enumerate(self.models):
			name = self.model_names[i]
			with self.subTest(f"ResNet model {name}", i=i):	
				watcher = ww.WeightWatcher(model=model, log_level=logging.WARNING)
				details = watcher.describe()

				M = details.M.to_numpy()
				N = details.N.to_numpy()
				self.assertTrue((N >= M).all())


	def test_num_evals(self):
		for i, model in enumerate(self.models):
			name = self.model_names[i]
			with self.subTest(f"ResNet model {name}", i=i):	
				watcher = ww.WeightWatcher(model=model, log_level=logging.WARNING)
				details = watcher.describe()	
					
				self.assertTrue((details.M * details.rf == details.num_evals).all())
              

		
class Test_RMT_Util(Test_Base):
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_RMT_Util:", self._testMethodName)
		
			
	def test_vector_entropy(self):
		u = np.array([1,1,1,1])
		actual_s = RMT_Util.vector_entropy(u)
		expected_s = 0.14676266317373993
		self.assertAlmostEqual(actual_s, expected_s,  places=6)
	
	def test_permute_matrix(self):
		W = 2.0*np.arange(20).reshape(4,5)
		p_W, p_ids = RMT_Util.permute_matrix(W)
		unp_W = RMT_Util.unpermute_matrix(p_W, p_ids)
		
		np.testing.assert_array_equal(W, unp_W)
		
	def test_detX_constraint(self):
		
		evals = np.zeros(100)
		evals[-10:]=1
		detX_num, detX_idx = RMT_Util.detX_constraint(evals,rescale=False)
		self.assertEqual(11, detX_num, "detX num")
		self.assertEqual(89, detX_idx, "detX idx")
		


	def test_combine_weights_and_biases(self):
		"""Test that we can combone W and b, stacked either way possible"""
		W = np.ones((2,3))
		b = 2*np.ones(3)
		expected_shape = (3,3)

		Wb = RMT_Util.combine_weights_and_biases(W,b)	
		actual_shape = Wb.shape
		self.assertEqual(expected_shape, actual_shape)

		W = np.ones((2,3))
		b = 2*np.ones(2)
		expected_shape = (2,4)

		Wb = RMT_Util.combine_weights_and_biases(W,b)	
		actual_shape = Wb.shape
		self.assertEqual(expected_shape, actual_shape)
		
		
	def test_line_entropy(self):
		data = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
		actual = RMT_Util.line_entropy(data)
		expected = 0.0
		self.assertAlmostEqual(actual, expected,  places=6)
		
		data = np.array([1,1,1,1,1,0,0,1,1,1,])
		actual = RMT_Util.line_entropy(data)
		expected = 0.5
		self.assertAlmostEqual(actual, expected,  places=3)


class Test_Vector_Metrics(Test_Base):
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_Vector_Metrics:", self._testMethodName)

	
	def test_valid_vectors(self):
		watcher = ww.WeightWatcher()

		vectors = None
		valid = watcher.valid_vectors(vectors)
		self.assertFalse(valid)

		vectors = 1
		valid = watcher.valid_vectors(vectors)
		self.assertFalse(valid)


		vectors = np.array([0,1,2,3,4])
		valid = watcher.valid_vectors(vectors)
		self.assertTrue(valid)

		vectors = [np.array([0,1,2,3,4]),np.array([0,1,2,3,4]),np.array([0,1,2,3,4])]
		valid = watcher.valid_vectors(vectors)
		self.assertTrue(valid)

		vectors = np.array([[0,1,2,3,4],[0,1,2,3,4]])
		valid = watcher.valid_vectors(vectors)
		self.assertTrue(valid)

	def test_vector_iterator(self):
		watcher = ww.WeightWatcher()

		vectors = np.array([0,1,2,3,4])
		iterator =  watcher.iterate_vectors(vectors)
		for num, v in enumerate(iterator):
			print("v= ",v)
		self.assertEqual(1, num+1)

		vectors = np.array([[0,1,2,3,4],[0,1,2,3,4]])
		iterator =  watcher.iterate_vectors(vectors)
		for num, v in enumerate(iterator):
			print("v= ",v)
		self.assertEqual(2, num+1)

		vectors = [np.array([0,1,2,3,4]),np.array([0,1,2,3,4]),np.array([0,1,2,3,4])]
		iterator =  watcher.iterate_vectors(vectors)
		for num, v in enumerate(iterator):
			print("v= ",v)
		self.assertEqual(3, num+1)

	def test_vector_metrics(self):
		watcher = ww.WeightWatcher()

		vectors = np.array([0,1,2,3,4])
		metrics = watcher.vector_metrics(vectors)
		print(metrics)


class Test_Distances(Test_Base):
	"""If we ever implement the idea of combining the biases into,  W+b_>W', then this class will 
		contain the unit tests for this.  
	"""
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_Distances:", self._testMethodName)
		
	def get_weights_and_biases_from_Keras(self):
		"""Test that we can get both weights and biases from pyTorch models"""
		
		ilayer_id = 21

		model = VGG16()
		print(type(model))
		watcher = ww.WeightWatcher(model=model, log_level=logging.WARNING)
		
		details = watcher.describe(layers=[21])
		print(details)
		
		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		
		params = ww.DEFAULT_PARAMS.copy()
		params[ww.ADD_BIASES] = True
		
		weights = watcher.get_Weights(layer=ilayer_id, params=params)
		self.assertEqual(len(weights),1)
		
		W = weights[0]
		self.assertEqual(np.max(W.shape),N)
		self.assertEqual(np.min(W.shape),M)

		pass
	
	
	def get_weights_and_biases_from_pyTorch(self):
		"""Test that we can get both weights and biases from Keras models"""
		
		ilayer_id = 28

		model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		watcher = ww.WeightWatcher(model = model, log_level=logging.WARNING)
		
		details = watcher.describe(layers=self.fclayers)
		print(details)
		
		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		
		params = ww.DEFAULT_PARAMS.copy()
		params[ww.ADD_BIASES] = True
		
		weights = watcher.get_Weights(layer=ilayer_id, params=params)
		self.assertEqual(len(weights),1)
		
		W = weights[0]
		self.assertEqual(np.max(W.shape),N)
		self.assertEqual(np.min(W.shape),M)

		pass
	
	def get_weights_and_biases_from_Onnx(self):
		"""Test that we can get both weights and biases from ONNX models"""
		
		pass




class Test_PyTorchSVD(Test_Base):
	"""
	Tests for discrepancies between the scipy and torch implementations of SVD.
	"""

	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_PyTorchSVD:", self._testMethodName)
		self.fc1_layer = 25
		self.fc2_layer = 28
		self.fc3_layer = 31

		self.fc_layers = [self.fc1_layer, self.fc2_layer, self.fc3_layer]


	def test_torch_svd(self):
		if RMT_Util._svd_full_fast is RMT_Util._svd_full_accurate:
			print("Warning: Unable to test PyTorch SVD method because torch / CUDA not available")
			return

		model = models.vgg11(weights='VGG11_Weights.IMAGENET1K_V1')
		watcher = ww.WeightWatcher(model = model, log_level=logging.WARNING)

		from time import time
		start = time()
		details_fast = watcher.analyze(layers=self.fc_layers, svd_method="fast")
		print(f"PyTorch (fast): {time() - start}s")
		start = time()
		details_accurate = watcher.analyze(layers=self.fc_layers, svd_method="accurate")
		print(f"SciPy (accurate): {time() - start}s")

		for f in ["alpha", "alpha_weighted", "D", "sigma", "sv_max", "xmin"]:
			self.assertLess(np.max(np.abs(details_fast.alpha - details_accurate.alpha)), 0.0025)

		for f in ["spectral_norm", "stable_rank", "xmax",]:
			self.assertLess(np.max(np.abs(details_fast.alpha - details_accurate.alpha)), 0.02)
			
			
	def test_bad_methods(self):
		W = np.ones((3,3))

		with self.assertRaises(AssertionError, msg="eig_full accepted a bad method"):
			RMT_Util.eig_full(W, "BAD_METHOD")

		with self.assertRaises(AssertionError, msg="svd_full accepted a bad method"):
			RMT_Util.svd_full(W, "BAD_METHOD")

		with self.assertRaises(AssertionError, msg="svd_vals accepted a bad method"):
			RMT_Util.svd_vals(W, "BAD_METHOD")

	def torch_available(self):
		try:
			from torch.cuda import is_available
			return is_available()
		except ImportError:
			return False

	def test_torch_availability(self):
		if self.torch_available():
			print("torch is available and cuda is available")
			self.assertFalse(RMT_Util._eig_full_accurate is RMT_Util._eig_full_fast)
			self.assertFalse(RMT_Util._svd_full_accurate is RMT_Util._svd_full_fast)
			self.assertFalse(RMT_Util._svd_vals_accurate is RMT_Util._svd_vals_fast)
		else:
			print("torch is not available or cuda is not available")
			self.assertTrue(RMT_Util._eig_full_accurate is RMT_Util._eig_full_fast)
			self.assertTrue(RMT_Util._svd_full_accurate is RMT_Util._svd_full_fast)
			self.assertTrue(RMT_Util._svd_vals_accurate is RMT_Util._svd_vals_fast)
			

	def test_torch_linalg(self):
		# Note that if torch is not available then this will test scipy instead.
		W = np.random.random((50,50))
		L, V = RMT_Util._eig_full_fast(W)
		W_reconstruct = np.matmul(V, np.matmul(np.diag(L), np.linalg.inv(V)))
		err = np.sum(np.abs(W - W_reconstruct))
		self.assertLess(err, 0.002, f"torch eig absolute reconstruction error was {err}")

		W = np.random.random((50,100))
		U, S, Vh = RMT_Util._svd_full_fast(W)
		W_reconstruct = np.matmul(U, np.matmul(np.diag(S), Vh[:50,:]))
		err = np.sum(np.abs(W - W_reconstruct))
		self.assertLess(err, 0.05, f"torch svd absolute reconstruction error was {err}")

		S_vals_only = RMT_Util._svd_vals_accurate(W)
		err = np.sum(np.abs(S - S_vals_only))
		self.assertLess(err, 0.0005, msg=f"torch svd and svd_vals differed by {err}")


class Test_PowerLaw(Test_Base):
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_PowerLaw:", self._testMethodName)
		self.model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)
		params = DEFAULT_PARAMS.copy()
		params[SVD_METHOD] = FAST_SVD
		self.evals = self.watcher.get_ESD(layer=67, params=params)


	def _check_fit_attributes(self, fit):
		types = {
			"alpha"               : (np.float64, ),
			"alphas"              : (np.ndarray, ),
			"D"                   : (np.float64, ),
			"Ds"                  : (np.ndarray, ),
			"sigma"               : (np.float64, ),
			"sigmas"              : (np.ndarray, ),
			"xmin"                : (np.float32, np.float64, ),
			"xmins"               : (np.ndarray, ),
			"xmax"                : (np.float32, np.float64, float, type(None)),
			"distribution_compare":	(type(self.setUp), ),	#i.e., function
			"plot_pdf"            : (type(self.setUp), ),	#i.e., function
		}
		for field, the_types in types.items():
			self.assertTrue(hasattr(fit, field), msg=f"fit object does not have attribute {field}")
			val = getattr(fit, field)
			self.assertTrue(isinstance(val, the_types), msg=f"fit.{field} was type {type(val)}")

	def _check_fit_attribute_len(self, fit, evals):
		M = len(evals)
		self.assertEqual(len(fit.alphas), M-1)
		self.assertEqual(len(fit.sigmas), M-1)
		self.assertEqual(len(fit.xmins), M-1)
		self.assertEqual(len(fit.Ds), M-1)

	def _check_fit_optimality(self, fit):
		i = np.argmin(fit.Ds)
		self.assertEqual(fit.alpha, fit.alphas[i])
		self.assertEqual(fit.sigma, fit.sigmas[i])
		self.assertEqual(fit.D, fit.Ds[i])

	def test_PL_speed(self):
		"""
		If the re-implementation of powerlaw is not faster then we should switch back.
		"""
		from time import time
		start_time = time()
		_ = WW_powerlaw.powerlaw.Fit(self.evals, distribution=POWER_LAW)
		PL_time = time() - start_time

		start_time = time()
		_ = WW_powerlaw.Fit(self.evals, distribution=POWER_LAW)
		WW_time = time() - start_time
		print(f"WW powerlaw time is {PL_time / WW_time:0.02f}x faster with M = {len(self.evals)}")

		self.assertLess(WW_time, PL_time)


	def test_fit_attributes(self):
		fit = WW_powerlaw.Fit(self.evals, distribution=POWER_LAW)
		self.assertIsInstance(fit, WW_powerlaw.Fit, msg=f"fit was {type(fit)}")

		self.assertAlmostEqual(fit.alpha, 3.3835113, places=3)
		self.assertAlmostEqual(fit.sigma, 0.3103067, places=4)
		self.assertAlmostEqual(fit.D, 	  0.0266789, places=4)

		self._check_fit_attributes(fit)

		self._check_fit_attribute_len(fit, self.evals)

		self._check_fit_optimality(fit)

	def test_fit_clipped_powerlaw(self):
		fit, num_fingers = RMT_Util.fit_clipped_powerlaw(self.evals)

		self.assertEqual(num_fingers, 0, msg=f"num_fingers was {num_fingers}")

		self._check_fit_attributes(fit)

		self._check_fit_attribute_len(fit, self.evals)

		self._check_fit_optimality(fit)

	
	def test_ww_power_law_fit_directly(self):
		"""We should have 1 or more tests that doesn't require ResNet evls incase these get corrupted
		
		We juit need to create some fake data that follows a Pareto distriibution"""

		def test_ww_power_law_fit_directly(self):
			"""We should have 1 or more tests that doesn't require ResNet evls incase these get corrupted"""

			np.random.seed(123)
			data = np.random.pareto(2.5, 100)
			
		
			result = WW_powerlaw.pl_fit(data, xmax=np.max(data), pl_package=POWERLAW_PACKAGE)
			expected_alpha = result.alpha
			self.assertAlmostEqual(expected_alpha, 2.5, delta=0.1)
	
			result = WW_powerlaw.pl_fit(data, xmax=np.max(data), pl_package=WW_POWERLAW_PACKAGE)
			actual_alpha = result.alpha	
			self.assertAlmostEqual(expected_alpha, actual_alpha, delta=0.1)
		
		

class Test_Plots(Test_Base):
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_Plots:", self._testMethodName)
		self.model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)

	def testPlots(self):
		""" Simply tests that the plot functions will not generate an exception.
			Does not guarantee correctness, yet.
		"""
		self.watcher.analyze(layers=[67], plot=True, randomize=True)

class Test_Pandas(Test_Base):
	def setUp(self):
		"""I run before every test in this class
		"""
		print("\n-------------------------------------\nIn Test_Pandas:", self._testMethodName)
		self.model = models.resnet18()
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.WARNING)

	def test_column_names_describe_basic(self):
		expected_columns = ['layer_id', 'name', 'M', 'N', 'Q', 'layer_type', 'longname',
							'num_evals', 'rf']

		details = self.watcher.describe(layers=[67])
		self.assertTrue(isinstance(details, pd.DataFrame), "details is a pandas DataFrame")
		self.assertEqual(len(expected_columns), len(details.columns))
		self.assertEqual(expected_columns, list(details.columns))

	def test_column_names_analyze(self):
		expected_columns = ['layer_id', 'name', 'D', 'Lambda', 'M', 'N', 'Q', 'alpha',
							'alpha_weighted', 'best_fit', 'entropy', 'fit_entropy', 'has_esd',
							'lambda_max', 'layer_type', 'log_alpha_norm', 'log_norm',
							'log_spectral_norm', 'longname', 'matrix_rank', 'norm', 'num_evals',
							'num_fingers', 'num_pl_spikes', 'rank_loss', 'rf', 'sigma',
							'spectral_norm', 'stable_rank', 'sv_max', 'warning', 'weak_rank_loss',
							'xmax', 'xmin']

		details = self.watcher.analyze(layers=[67])
		self.assertTrue(isinstance(details, pd.DataFrame), "details is a pandas DataFrame")
		self.assertEqual(len(expected_columns), len(details.columns))
		self.assertEqual(expected_columns, list(details.columns))

	def test_column_names_analyze_detX(self):
		expected_columns = ['layer_id', 'name', 'D', 'Lambda', 'M', 'N', 'Q', 'alpha',
							'alpha_weighted', 'best_fit', 'detX_num', 'detX_val',
							'detX_val_unrescaled', 'entropy', 'fit_entropy', 'has_esd',
							'lambda_max', 'layer_type', 'log_alpha_norm', 'log_norm',
							'log_spectral_norm', 'longname', 'matrix_rank', 'norm', 'num_evals',
							'num_fingers', 'num_pl_spikes', 'rank_loss', 'rf', 'sigma',
							'spectral_norm', 'stable_rank', 'sv_max', 'warning', 'weak_rank_loss',
							'xmax', 'xmin']

		details = self.watcher.analyze(layers=[67], detX=True)
		self.assertTrue(isinstance(details, pd.DataFrame), "details is a pandas DataFrame")
		self.assertEqual(len(expected_columns), len(details.columns))
		self.assertEqual(expected_columns, list(details.columns))

	def test_column_names_analyze_randomize(self):
		expected_columns = ['layer_id', 'name', 'D', 'Lambda', 'M', 'N', 'Q', 'alpha',
 					        'alpha_weighted', 'best_fit', 'entropy', 'fit_entropy', 'has_esd',
					        'lambda_max', 'layer_type', 'log_alpha_norm', 'log_norm',
					        'log_spectral_norm', 'longname', 'matrix_rank', 'max_rand_eval', 'norm',
					        'num_evals', 'num_fingers', 'num_pl_spikes', 'rand_W_scale',
					        'rand_bulk_max', 'rand_bulk_min', 'rand_distance', 'rand_mp_softrank',
					        'rand_num_spikes', 'rand_sigma_mp', 'rank_loss', 'rf', 'sigma',
					        'spectral_norm', 'stable_rank', 'sv_max', 'warning', 'weak_rank_loss',
					        'ww_maxdist', 'ww_softrank', 'xmax', 'xmin']

		details = self.watcher.analyze(layers=[67], randomize=True)
		self.assertTrue(isinstance(details, pd.DataFrame), "details is a pandas DataFrame")
		self.assertEqual(len(expected_columns), len(details.columns))
		self.assertEqual(expected_columns, list(details.columns))

	def test_column_names_analyze_intra(self):
		expected_columns = ['layer_id', 'name', 'D', 'Lambda', 'M', 'N', 'Q', 'Xflag', 'alpha',
					        'alpha_weighted', 'best_fit', 'entropy', 'fit_entropy', 'has_esd',
					        'lambda_max', 'layer_type', 'log_alpha_norm', 'log_norm',
					        'log_spectral_norm', 'longname', 'matrix_rank', 'norm', 'num_evals',
					        'num_fingers', 'num_pl_spikes', 'rank_loss', 'rf', 'sigma',
					        'spectral_norm', 'stable_rank', 'sv_max', 'warning', 'weak_rank_loss',
					        'xmax', 'xmin']

		details = self.watcher.analyze(layers=[64, 67], intra=True)
		self.assertTrue(isinstance(details, pd.DataFrame), "details is a pandas DataFrame")
		self.assertEqual(len(expected_columns), len(details.columns))
		self.assertEqual(expected_columns, list(details.columns))

	def test_column_names_analyze_mp_fit(self):
		expected_columns = ['layer_id', 'name', 'D', 'Lambda', 'M', 'N', 'Q', 'W_scale', 'alpha',
							'alpha_weighted', 'best_fit', 'bulk_max', 'bulk_min', 'entropy',
							'fit_entropy', 'has_esd', 'lambda_max', 'layer_type', 'log_alpha_norm',
							'log_norm', 'log_spectral_norm', 'longname', 'matrix_rank',
							'mp_softrank', 'norm', 'num_evals', 'num_fingers', 'num_pl_spikes',
							'num_spikes', 'rank_loss', 'rf', 'sigma', 'sigma_mp', 'spectral_norm',
							'stable_rank', 'sv_max', 'warning', 'weak_rank_loss', 'xmax', 'xmin']

		details = self.watcher.analyze(layers=[67], mp_fit=True)
		self.assertTrue(isinstance(details, pd.DataFrame), "details is a pandas DataFrame")
		self.assertEqual(len(expected_columns), len(details.columns))
		self.assertEqual(expected_columns, list(details.columns))




		
		
		
if __name__ == '__main__':
	unittest.main()
