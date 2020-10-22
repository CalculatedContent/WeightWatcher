import unittest
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import sys, logging
import weightwatcher as ww
from weightwatcher import  LAYER_TYPE 

import torchvision.models as models
import pandas as pd


class Test_VGG11(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		"""I run only once for this class
		"""

		cls.model = models.vgg11(pretrained=True)
		cls.watcher = ww.WeightWatcher(model=cls.model, log=False)
		logging.getLogger("weightwatcher").setLevel(logging.INFO)
		
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
			
			
	def test_all_columns(self):
		"""Test that new results are returns a valid pandas dataframe
		"""
 		
		details = self.watcher.analyze()
		self.assertEqual(isinstance(details, pd.DataFrame), True, "details is a pandas DataFrame")
 
		
		columns = "layer_id,name,D,M,N,alpha,alpha_weighted,has_esd,lambda_max,layer_type,log_alpha_norm,log_norm,log_spectral_norm,norm,num_evals,rank_loss,rf,sigma,spectral_norm,stable_rank,sv_max,xmax,xmin".split(',')

		for key in columns:
			self.assertTrue(key in details.columns, "{} in details. Columns are {}".format(key, details.columns))
 
 
	def test_model_layer_types_ww2x(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers"
		"""
 
		details = self.watcher.describe(ww2x=True)
		print(details)
		
		denseLayers = details[details.layer_type==str(LAYER_TYPE.DENSE)]
		denseCount = len(denseLayers)
		self.assertEquals(denseCount, 3, "3 dense layers, but {} found".format(denseCount))
 			
	
		conv2DLayers = details[details.layer_type==str(LAYER_TYPE.CONV2D)]
		conv2DCount = len(conv2DLayers)
		self.assertEquals(conv2DCount, 8*9, "8*9 conv2D layers, but {} found".format(denseCount))
	
	def test_all_layer_types(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers"
		"""

		details = self.watcher.describe()
		print(details)
		
		denseLayers = details[details.layer_type==str(LAYER_TYPE.DENSE)]
		denseCount = len(denseLayers)
		self.assertEquals(denseCount, 3, "3 dense layers, but {} found".format(denseCount))		
	
		conv2DLayers = details[details.layer_type==str(LAYER_TYPE.CONV2D)]
		conv2DCount = len(conv2DLayers)
		self.assertEquals(conv2DCount, 8, "8 conv2D layers, but {} found".format(denseCount))
	
		
	def test_filter_dense_layer_types(self):
		"""Test that ww.LAYER_TYPE.DENSE filter is applied only to DENSE layers"
		"""
		print("test_filter_dense_layer_types")
		details = self.watcher.describe(layers=[LAYER_TYPE.DENSE])
		print(details)
		
		denseLayers = details[details.layer_type==str(LAYER_TYPE.DENSE)]
		denseCount = len(denseLayers)

		self.assertEquals(denseCount, 3, "3 dense layers, but {} found".format(denseCount))
			
		# Dense layers are analyzed
		self.assertTrue((denseLayers.N > 0).all(axis=None), "All {} dense layers have a non zero N".format(denseCount))
		self.assertTrue((denseLayers.M > 0).all(axis=None), "All {} dense layers have a non zero M".format(denseCount))

		nonDenseLayers = details[details.layer_type!=str(LAYER_TYPE.DENSE)]
		nonDenseCount = len(nonDenseLayers)

		self.assertEquals(nonDenseCount, 0, "Filter has No dense layers: {} found".format(nonDenseCount))

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
		self.assertEquals(denseCount, 3, "3 dense layers, but {} found".format(denseCount))
			
		nonDenseLayers = details[details.layer_type!=str(LAYER_TYPE.DENSE)]
		nonDenseCount = len(nonDenseLayers)
		self.assertEquals(nonDenseCount, 0, "Filter has No dense layers: {} found".format(nonDenseCount))


	def test_filter_conv2D_layer_types(self):
		"""Test that ww.LAYER_TYPE.CONV2D filter is applied only to CONV2D layers"
		"""

		details = self.watcher.describe(layers=ww.LAYER_TYPE.CONV2D)
		print(details)

		conv2DLayers = details[details['layer_type']==str(LAYER_TYPE.CONV2D)]
		conv2DCount = len(conv2DLayers)
		self.assertEquals(conv2DCount, 8, "# conv2D layers: {} found".format(conv2DCount))
		nonConv2DLayers = details[details['layer_type']!=str(LAYER_TYPE.CONV2D)]
		nonConv2DCount = len(nonConv2DLayers)
		self.assertEquals(nonConv2DCount, 0, "VGG11 has non conv2D layers: {} found".format(nonConv2DCount))

	

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
		print(details)
		self.assertEqual(len(details), 75)
		
 		
	def test_compute_alphas(self):
		"""Test that alphas are computed and values are within thresholds
		"""
		details = self.watcher.analyze(layers=[5], ww2x=True, randomize=False, plot=False, mp_fit=False)
		print("test_compute_alphas:  details:")
		print(details[['layer_id', 'alpha']])
		#d = self.watcher.get_details(results=results)
		a = details.alpha.to_numpy()
		self.assertAlmostEqual(a[0],1.65014, places=4)
		self.assertAlmostEqual(a[1],1.57297, places=4)
		self.assertAlmostEqual(a[3],1.43459, places=4)
 		
		# spectral norm
		a = details.spectral_norm.to_numpy()
		print("------------FIX THIS--------------")
		print(a[0], a[1], a[2])
		#self.assertAlmostEqual(a[0],20.2149, places=4)
		#self.assertAlmostEqual(a[1],24.8158, places=4)
		#self.assertAlmostEqual(a[2],19.3795, places=4)
		
		

# TODO: test the input option match older options and work properly
# 
# 	def test_normalize(self):
# 		"""Test that weight matrices are normalized as expected
# 		"""
# 

	def test_getESD(self):
		"""Test that eigenvalues are available 
		"""

		esd = self.watcher.get_ESD(layer=5)
		self.assertEqual(len(esd), 576)


	def test_density_fit(self):
		"""Test the fitted sigma from the density fit
		"""
 		
		print("----test_density_fit-----")
		print("PLOT STILL PLOTTING..FIX")
		details = self.watcher.analyze(layers = [10], randomize=False, plot=False, mp_fit=True)
		print(details)
 
		#df = df.reset_index()
		#self.assertAlmostEqual(df.loc[0, 'sigma_mp'], 1.00, places=2) #sigma_mp
		#self.assertAlmostEqual(df.loc[0, 'numofSpikes'], 13) #numofSig
		#self.assertAlmostEqual(df.loc[0, 'sigma_mp'], 1.064648437) #sigma_mp
		#self.assertAlmostEqual(df.loc[0, 'numofSpikes'], 30.00) #numofSig
		#self.assertAlmostEqual(df.loc[0, 'ratio_numofSpikes'], 0.117647, places = 6)
		#self.assertAlmostEqual(df.loc[0, 'softrank_mp'], 0.203082, places = 6)
# 


	def test_runtime_warnings(self):
		"""Test that runtime warnings are still active
		"""
		import numpy as np
		print("test runtime warning: sqrt(-1)=", np.sqrt(-1.0))
		assert(True)
		


if __name__ == '__main__':
	unittest.main()
