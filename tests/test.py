import unittest
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import sys, logging
import weightwatcher as ww
from weightwatcher import  LAYER_TYPE 

import torchvision.models as models
import pandas as pd

from transformers import TFAutoModelForSequenceClassification

#  https://kapeli.com/cheat_sheets/Python_unittest_Assertions.docset/Contents/Resources/Documents/index

class Test_VGG11(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		"""I run only once for this class
		"""

		cls.model = models.vgg11(pretrained=True)
		cls.watcher = ww.WeightWatcher(model=cls.model, log_level=logging.DEBUG)
		#logging.getLogger("weightwatcher").setLevel(logging.INFO)
		
	def setUp(self):
		"""I run before every test in this class
		"""
		pass

# 
# 	def test_summary_is_dict(self):
# 		"""Test that get_summary() returns a valid python dict
# 		"""
# 		self.watcher.analyze()
# 		summary = self.watcher.get_summary()
# 
# 		self.assertTrue(isinstance(summary, dict), "Summary is a dictionary")
# 
# 		for key in ['norm', 'norm_compound', 'lognorm', 'lognorm_compound']:
# 			self.assertTrue(key in summary, "{} in summary".format(key))
# 
# 
	def test_basic_columns(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
		
		details = self.watcher.describe()
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")
		print(details)
		# TODO: add more columns ?
		for key in ['layer_id', 'name', 'M', 'N']:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))
			
			
	def test_analyze_columns(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
	
		details = self.watcher.analyze()
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")

		
		columns = "layer_id,name,D,M,N,alpha,alpha_weighted,has_esd,lambda_max,layer_type,log_alpha_norm,log_norm,log_spectral_norm,norm,num_evals,rank_loss,rf,sigma,spectral_norm,stable_rank,sv_max,xmax,xmin,num_pl_spikes".split(',')

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
						
		# ww2x and conv2d_fft 
		with self.assertRaises(Exception) as context:
			self.watcher.describe(ww2x=True, conv2d_fft=True)	

		# intra and conv2d_fft
		with self.assertRaises(Exception) as context:
			self.watcher.describe(intra=True, conv2d_fft=True)	
				
				
		# min_evals > max_evals
		with self.assertRaises(Exception) as context:
			self.watcher.describe(min_evals=100, max_evals=10)	
 
	def test_model_layer_types_ww2x(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers"
		"""
 
		details = self.watcher.describe(ww2x=True)
		
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
		
		details = self.watcher.describe(layers=[25,28,31])
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
		
		details = self.watcher.describe(layers=[])
		print(details)
		
		details = self.watcher.describe(layers=[-25,-28,-31])
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

		print("test_min_matrix_shape")
		details = self.watcher.describe(min_evals=30)
		print(details)

		for nev in details.num_evals:
			self.assertGreaterEqual(nev, 30)
		

	def test_max_matrix_shape(self):
		"""Test that analyzes skips matrices larger than  MAX matrix shape
		"""

		print("test_max_matrix_shape")
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
		"""
		details = self.watcher.describe(ww2x=True)
		self.assertEqual(len(details), 75)
		
		
	def test_switch_channels(self):
		"""Test user can switch the channels for a Conv2D layer
		"""
		details = self.watcher.describe(layers=[2],  channels='first')
		N = details.N.to_numpy()[0]
		M = details.M.to_numpy()[0]
		rf = details.rf.to_numpy()[0]
		num_evals = details.num_evals.to_numpy()[0]
		
		self.assertEqual(N, 3)
		self.assertEqual(M, 3)
		self.assertEqual(rf, 3*64)
		self.assertEqual(num_evals, 3*3*64)

 		
	def test_compute_alphas(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		details = self.watcher.analyze(layers=[5], ww2x=True, randomize=False, plot=False, mp_fit=False)
		#d = self.watcher.get_details(results=results)
		a = details.alpha.to_numpy()
		self.assertAlmostEqual(a[0],1.65014, places=4)
		self.assertAlmostEqual(a[1],1.57297, places=4)
		self.assertAlmostEqual(a[3],1.43459, places=4)
 		
		# spectral norm
		a = details.spectral_norm.to_numpy()
		self.assertAlmostEqual(a[0],20.2149, places=4)
		self.assertAlmostEqual(a[1],24.8158, places=4)
		self.assertAlmostEqual(a[2],19.3795, places=4)
		
		
	def test_get_details(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		actual_details = self.watcher.analyze(layers=[5])
		expected_details = self.watcher.get_details()
		
		self.assertEqual(len(actual_details), len(expected_details), "actual and expected details differ")
		
	def test_get_summary(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		details = self.watcher.analyze(layers=[5])
		returned_summary = self.watcher.get_summary(details)
		
		print(returned_summary)
		
		saved_summary = self.watcher.get_summary()
		self.assertEqual(returned_summary, saved_summary)


	def test_getESD(self):
		"""Test that eigenvalues are available 
		"""

		esd = self.watcher.get_ESD(layer=5)
		self.assertEqual(len(esd), 576)


	def randomize(self):
		"""Test randomize option
		"""
		
		print("----test_density_fit-----")
		details = self.watcher.analyze(layers = [25], randomize=False, plot=False, mp_fit=False)
		print(details.columns)
		self.assertNotIn('max_rand_eval', details.columns)
		
		print("----test_density_fit-----")
		details = self.watcher.analyze(layers = [25], randomize=True, plot=False, mp_fit=False)
		print(details.columns)
		self.assertIn('max_rand_eval', details.columns)
		
		
		
	
	def test_density_fit(self):
		"""Test the fitted sigma from the density fit
		"""
 		
		print("----test_density_fit-----")
		details = self.watcher.analyze(layers = [25], ww2x=True, randomize=False, plot=False, mp_fit=True)
		print(details.columns)
		print("num spikes", details.num_spikes)
		print("sigma mp", details.sigma_mp)
		print("softrank", details.mp_softrank)

		#self.assertAlmostEqual(details.num_spikes, 13) #numofSig
		#self.assertAlmostEqual(details.sigma_mp, 1.064648437) #sigma_mp
		#self.assertAlmostEqual(details.np_softrank, 0.203082, places = 6) 


	def test_runtime_warnings(self):
		"""Test that runtime warnings are still active
		"""
		import numpy as np
		print("test runtime warning: sqrt(-1)=", np.sqrt(-1.0))
		assert(True)
		

class Test_TFBert(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		"""I run only once for this class
		"""
		CHECKPOINT = "bert-base-uncased"
		cls.model = TFAutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
		cls.watcher = ww.WeightWatcher(model=cls.model, log_level=logging.DEBUG)
		
	def setUp(self):
		"""I run before every test in this class
		"""
		pass

	def test_num_layers(self):
		"""Test that the Keras Iterator finds all the TFBert layers
		"""
		details = self.watcher.describe()
		print("TESTING TF BERT")
		self.assertTrue(len(details), 963)


if __name__ == '__main__':
	unittest.main()
