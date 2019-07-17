import unittest

import weightwatcher as ww

class Test_VGG11(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		"""I run only once for this class
		"""
		import torchvision.models as models
		cls.model = models.vgg16(pretrained=True)
		cls.watcher = ww.WeightWatcher(model=cls.model, log=False)


	def setUp(self):
		"""I run before every test in this class
		"""
		pass


	def test_summary_is_dict(self):
		self.watcher.analyze()
		summary = self.watcher.get_summary()

		self.assertTrue(isinstance(summary, dict), "Summary is a dictionary")

		for key in ['norm', 'norm_compound', 'lognorm', 'lognorm_compound']:
			self.assertTrue(key in summary, "{} in summary".format(key))


	def test_pandas(self):
		import pandas as pd

		self.watcher.analyze()
		summary = self.watcher.get_summary(pandas=True)

		self.assertEqual(isinstance(summary, pd.DataFrame), True, "Summary is a pandas DataFrame")

		columns = ",".join(summary.columns)
		for key in ['norm', 'norm_compound', 'lognorm', 'lognorm_compound']:
			self.assertTrue(key in summary.columns, "{} in summary. Columns are {}".format(key, columns))


	def test_filter_layer_types(self):
		import pandas as pd

		results = self.watcher.analyze(layers=ww.LAYER_TYPE.DENSE)
		d = self.watcher.get_details(results=results)

		denseLayers = d[d['layer_type']=="DENSE"]
		denseCount = len(denseLayers)

		self.assertTrue(denseCount > 0, "Non zero number of dense layers: {} found".format(denseCount))
			
		# Dense layers are analyzed
		self.assertTrue((denseLayers.N > 0).all, "All {} dense layers have a non zero N".format(denseCount))
		self.assertTrue((denseLayers.M > 0).all, "All {} dense layers have a non zero M".format(denseCount))

		nonDenseLayers = d[d['layer_type']!="DENSE"]
		nonDenseCount = len(nonDenseLayers)

		self.assertTrue(nonDenseCount > 0, "VGG16 has non dense layers: {} found".format(nonDenseCount))
		
		# Non Dense layers are NOT analyzed
		self.assertTrue((nonDenseLayers.N == 0).all, "All {} NON dense layers have a zero N".format(nonDenseCount))
		self.assertTrue((nonDenseLayers.M == 0).all, "All {} NON dense layers have a zero M".format(nonDenseCount))


if __name__ == '__main__':
	unittest.main()
