import unittest
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import sys, logging
import weightwatcher as ww
from weightwatcher import  LAYER_TYPE 
from weightwatcher import  DEFAULT_PARAMS 
from weightwatcher import  PL, TPL, E_TPL, POWER_LAW, TRUNCATED_POWER_LAW, LOG_NORMAL


import torchvision.models as models
import pandas as pd

from transformers import TFAutoModelForSequenceClassification

from tensorflow.keras.applications.vgg16 import VGG16

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

		# savefig is not a string or boolea
		with self.assertRaises(Exception) as context:
			self.watcher.describe(savefig=-1)	

		self.watcher.describe(savefig=True)
		self.watcher.describe(savefig='tmpdir')	
 
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



	def test_randomize(self):
		"""Test randomize option
		"""
		
		print("----test_density_fit-----")
		details = self.watcher.analyze(layers = [25], randomize=False, plot=False, mp_fit=False)
		print(details.columns)
		self.assertNotIn('max_rand_eval', details.columns)
		
		print("----test_density_fit-----")
		details = self.watcher.analyze(layers = [25], randomize=True, plot=False, mp_fit=False)
		print(details.rand_distance[0])
		self.assertIn('max_rand_eval', details.columns)
		
		
				
	def test_rand_distance(self):
		"""Test rand distance
		   Not very accuracte since it is random
		"""
		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model,  log_level=logging.DEBUG)
		details= self.watcher.analyze(layers=[28], randomize=True)
		actual = details.rand_distance[0]
		expected = 0.29
		self.assertAlmostEqual(actual,expected, places=2)
		
	def test_reset_params(self):
		"""test that params are reset / normalized ()"""
		
		params = DEFAULT_PARAMS
		params['fit']=PL
		valid = self.watcher.valid_params(params)
		self.assertTrue(valid)
		params = self.watcher.normalize_params(params)
		self.assertEqual(params['fit'], POWER_LAW)
		
		params = DEFAULT_PARAMS
		params['fit']=TPL
		valid = self.watcher.valid_params(params)
		self.assertTrue(valid)
		params = self.watcher.normalize_params(params)
		self.assertEqual(params['fit'], TRUNCATED_POWER_LAW)
		
		
	def test_intra_power_law_fit(self):
		"""Test PL fits on intra
		"""
		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model,  log_level=logging.DEBUG)
		details= self.watcher.analyze(layers=[25, 28],intra=True)
		actual_alpha = details.alpha[0]
		actual_best_fit = details.best_fit[0]
		print(actual_alpha,actual_best_fit)

		expected_alpha =  2.654 # not very accurate because of the sparisify transform
		expected_best_fit = LOG_NORMAL
		self.assertAlmostEqual(actual_alpha,expected_alpha, places=1)
		self.assertEqual(actual_best_fit, expected_best_fit)
		
		
	def test_intra_power_law_fit2(self):
		"""Test PL fits on intram, sparsify off, more accurate
		WHy does this fail ?
		"""
		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model,  log_level=logging.DEBUG)
		details= self.watcher.analyze(layers=[25, 28], intra=True, sparsify=False)
		actual_alpha = details.alpha[0]
		actual_best_fit = details.best_fit[0]
		print(actual_alpha,actual_best_fit)


		expected_alpha =  2.719
		expected_best_fit = LOG_NORMAL
		self.assertAlmostEqual(actual_alpha,expected_alpha, places=2)
		self.assertEqual(actual_best_fit, expected_best_fit)
					
	def test_truncated_power_law_fit(self):
		"""Test TPL fits
		"""
		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model,  log_level=logging.DEBUG)
		details= self.watcher.analyze(layers=[28], fit='TPL')
		actual_alpha = details.alpha[0]
		actual_Lambda = details.Lambda[0]

		self.assertTrue(actual_Lambda > -1) #Lambda must be set for TPL

		# these numbers have not been independently verified yet
		expected_alpha = 2.1075
		expected_Lambda =  0.01667
		self.assertAlmostEqual(actual_alpha,expected_alpha, places=3)
		self.assertAlmostEqual(actual_Lambda,expected_Lambda, places=3)
		
		
	def test_extended_truncated_power_law_fit(self):
		"""Test E-TPL fits.  Runs TPL with fix_fingets = XMIN_PEAK
		"""
		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model,  log_level=logging.DEBUG)
		details= self.watcher.analyze(layers=[28], fit=E_TPL)
		actual_alpha = details.alpha[0]
		actual_Lambda = details.Lambda[0]

		self.assertTrue(actual_Lambda > -1) #Lambda must be set for TPL
		
		# these numbers have not been independently verified yet
		expected_alpha = 2.07986
		expected_Lambda =  0.01983
		self.assertAlmostEqual(actual_alpha,expected_alpha, places=3)
		self.assertAlmostEqual(actual_Lambda,expected_Lambda, places=3)
		 
		
		
	def test_fix_fingers_xmin_peak(self):
		"""Test fix fingers xmin_peak 
		"""
		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model,  log_level=logging.DEBUG)
		# default
		details = self.watcher.analyze(layers=[5])
		actual = details.alpha.to_numpy()[0]
		expected = 7.116304
		print("ACTUAL {}".format(actual))
		self.assertAlmostEqual(actual,expected, places=4)
		
		# XMIN_PEAK
		details = self.watcher.analyze(layers=[5], fix_fingers='xmin_peak')
		actual = details.alpha[0]
		actual = details.alpha.to_numpy()[0]
		expected = 1.422195
		self.assertAlmostEqual(actual,expected, places=4)
		
		
	def test_fix_fingers_clip_xmax(self):
		"""Test fix fingers clip_xmax
		"""
		self.model = models.vgg11(pretrained=True)	
		
		# CLIP_XMAX
		details = self.watcher.analyze(layers=[5], fix_fingers='clip_xmax')
		actual = details.alpha.to_numpy()[0]
		expected = 1.663549
		self.assertAlmostEqual(actual,expected, places=4)
		


		
	
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


	def test_svd_smoothing(self):
		"""Test the svd smoothing on 1 lyaer of VGG
		"""
 		
		print("----test_svd_smoothing-----")

		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.DEBUG)
		
		# 819 =~ 4096*0.2
		self.watcher.SVDSmoothing(layers=[28])
		esd = self.watcher.get_ESD(layer=28) 
		num_comps = len(esd[esd>10**-10])
		self.assertEqual(num_comps, 819)

	def test_svd_smoothing_alt(self):
		"""Test the svd smoothing on 1 lyaer of VGG
		The intent is that test_svd_smoothing and test_svd_smoothing_lat are exactly the same
		except that:

		test_svd_smoothing() only applies TruncatedSVD, and can only keep the top N eigenvectors

		whereas

		test_svd_smoothing_alt() allows for a negative input, which throws away the top N eigenvectors

		Note:  I changed the APi on these method recently and that may be the bug
		this needs to be stabilzed for the ww.0.5 release
		"""
 		
		print("----test_svd_smoothing_alt-----")

		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.DEBUG)
		
		self.watcher.SVDSmoothing(layers=[28], percent=-0.2)
		esd = self.watcher.get_ESD(layer=28) 
		num_comps = len(esd[esd>10**-10])
		# 3277 = 4096 - 819
		print("num comps = {}".format(num_comps))
		self.assertEqual(num_comps, 3277)
		
	def test_svd_smoothing_alt2(self):
		"""Test the svd smoothing on 1 lyaer of VGG
		
		"""
 		
		print("----test_svd_smoothing_alt2-----")

		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.DEBUG)
		
		self.watcher.SVDSmoothing(layers=[28], percent=0.2)
		esd = self.watcher.get_ESD(layer=28) 
		num_comps = len(esd[esd>10**-10])
		# 3277 = 4096 - 819
		print("num comps = {}".format(num_comps))
		self.assertEqual(num_comps, 819)
		
		
		
	def test_svd_sharpness(self):
		"""Test the svd smoothing on 1 lyaer of VGG
		"""
 		
		print("----test_svd_sharpness-----")

		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.DEBUG)

		esd_before = self.watcher.get_ESD(layer=28) 
		
		self.watcher.SVDSharpness(layers=[28])
		esd_after = self.watcher.get_ESD(layer=28) 
		
		print("max esd before {}".format(np.max(esd_before)))
		print("max esd after {}".format(np.max(esd_after)))

		self.assertGreater(np.max(esd_before)-2.0,np.max(esd_after))
		
	
		
	def test_svd_sharpness2(self):
		"""Test the svd smoothing on 1 lyaer of VGG
		"""
 		
		print("----test_svd_sharpness-----")

		self.model = models.vgg11(pretrained=True)
		self.watcher = ww.WeightWatcher(model=self.model, log_level=logging.DEBUG)

		esd_before = self.watcher.get_ESD(layer=8) 
		
		self.watcher.SVDSharpness(layers=[8])
		esd_after = self.watcher.get_ESD(layer=8) 
		
		print("max esd before {}".format(np.max(esd_before)))
		print("max esd after {}".format(np.max(esd_after)))

		self.assertGreater(np.max(esd_before),np.max(esd_after))
		
	

	
		

	def test_runtime_warnings(self):
		"""Test that runtime warnings are still active
		"""
		import numpy as np
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
		details = self.watcher.analyze(mp_fit=True,  randomize=True,  ww2x=False)
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


		iterator = self.watcher.make_layer_iterator(model=self.model)
		num = 0
		actual_ids = []
		for ww_layer in iterator:
			self.assertGreater(ww_layer.layer_id,0)
			actual_ids.append(ww_layer.layer_id)
			num += 1
		self.assertEqual(num,11)
		self.assertEqual(actual_ids,expected_ids)

		
		iterator = self.watcher.make_layer_iterator(model=self.model, layers=[28])
		num = 0
		for ww_layer in iterator:
			self.assertEqual(ww_layer.layer_id,28)
			num += 1
		self.assertEqual(num,1)
		
		
	# CHM:  stacked layers may not be working properly, be careful
	# needs more testing 
	def test_ww_stacked_layer_iterator(self):
		"""Test Stacked Layer Iterator
		"""
				
		params = DEFAULT_PARAMS
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
		
		
				
	def test_permute_W(self):
		"""Test that permute and unpermute methods work
		"""
		N, M = 4096, 4096
		iterator = self.watcher.make_layer_iterator(model=self.model, layers=[28])
		for ww_layer in iterator:
			self.assertEqual(ww_layer.layer_id,28)
			W = ww_layer.Wmats[0]
			self.assertEqual(W.shape,(N,M))
			
			self.watcher.apply_permute_W(ww_layer)
			W2 = ww_layer.Wmats[0]
			self.assertNotEqual(W[0,0],W2[0,0])
			
			self.watcher.apply_unpermute_W(ww_layer)
			W2 = ww_layer.Wmats[0]
			self.assertEqual(W2.shape,(N,M))
			self.assertEqual(W[0,0],W2[0,0])
			

		

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
		72 layers for BERT
		+ 1 for input. 1 for output
		Not sure why it is 76
		"""
		details = self.watcher.describe()
		print("TESTING TF BERT")
		self.assertEqual(len(details), 74)




class Test_Keras(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		"""I run only once for this class
		"""
		cls.model = VGG16()
		cls.watcher = ww.WeightWatcher(model=cls.model, log_level=logging.DEBUG)
		
	def setUp(self):
		"""I run before every test in this class
		"""
		pass

	def test_num_layers(self):
		"""Test that the Keras on VGG11
		"""
		details = self.watcher.describe()
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


class Test_ResNet(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		"""I run only once for this class
		"""
		cls.model = models.resnet18(pretrained=True)
		cls.watcher = ww.WeightWatcher(model=cls.model, log_level=logging.DEBUG)
		
	def setUp(self):
		"""I run before every test in this class
		"""
		pass
	
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
		
		
	
from weightwatcher import RMT_Util
import numpy as np

class Test_RMT_Util(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		"""I run only once for this class
		"""
		
	def setUp(self):
		"""I run before every test in this class
		"""
		pass

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


if __name__ == '__main__':
	unittest.main()
