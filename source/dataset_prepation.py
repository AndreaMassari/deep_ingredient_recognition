"""
This module loads image and recipe data and stores features and labels in numpy arrays
"""
import numpy as np

class data_preparator:
	"""
	This class loads image and recipe data and stores features and labels in numpy arrays
	"""
	def __init__(self):
		self.numb_ing=
		self.ingred_freq=np.zeros(self.numb_ing)
		self.image_labels={}
		return

	def data_loader(self):
		"""
		loading the json dictionaries saved in preprocessing
		"""
		with open(homedir+'/ingdict.json') as f:
			self.ingdict=json.load(f)
		#ingdict=['basil','beans','beef','beer','carrots','chicken','clams',... 
		#these are the ingredients that do show up in the dataset, out of our ing_list
		with open(homedir+'/catgdict.json') as f:
			self.catgdict=json.load(f)
		#catgdict=["ARGO®, KARO®, FLEISCHMANN'S®",'Acorn Squash Side Dishes','Adult Punch','African','Almond Board',
		#'Almond Desserts','Amaretto',... 
		#these are the food categories that show up in the dataset, notice that the generic category "Recipes" 
		#covers a large fractions of the recipes, making these data points not useful for the multitasking classification
		with open(homedir+'/list_rec_eff.json') as f:
			self.list_rec_eff=json.load(f)
		#list_rec_eff=['100008','100011','100033','100056','100057','100058','100059','100060',...
		#recipe ids that correspond to at least one image
		with open(homedir+'/ingrotot.json') as f:
			self.ingrotot=json.load(f)  
		#ingrotot=[{'category': 'Pickled','id': '100008','ingredients': ['rice'],'name': 'Homemade Pickled Ginger (Gari)'},
		#          {'category': 'Pork','id': '100011','ingredients': ['chicken', 'pork', 'garlic', 'beans', 'onion'],
		#          'name': 'Pork and Black Bean Stew'},{'category': 'Frostings and Icings','id': '100033','ingredients': 
		#           ['cream'],'name': 'Honey-Cocoa Frosting'},...
		#list of recipe labels, each one a dictionary with keys: "category", "id", "ingredients", and "name" of the recipe
		ingrototret = []
		with open(homedir+'/ingrototret.json') as f:
			self.ingrototret=json.load(f)  
		#ingrototret={'150893': ['Johnnie Walker Gold Ice Gold', 'Recipes', []],
		#             '118167': ['Apple Sausage Ring','Recipes', ['eggs', 'onion', 'pork', 'milk']],
		#            '117576': ['Vegetable Bean Barley Soup','Recipes',['basil','tomatoes','beans','onion','chicken','pork',
		#             'garlic','carrots']],...
		#same as before except this is a dictionary with recipe ids as keys, for convenience 
		# here I make the string category/ingredients labels into numerical classes with these handy dictionaries
		self.classdict_ing={ii:ingdict.index(ii) for ii in ingdict}
		self.classdict_ing_rev={ii:ingdict[ii] for ii in range(len(ingdict))}
		self.classdict_cat={ii:catgdict.index(ii) for ii in catgdict}
		self.classdict_cat_rev={ii:catgdict[ii] for ii in range(len(catgdict))}
		#these are identical to ingrototret, except again with numerical labels, instead of strings
		self.ingdictlist={jj["id"]:[ingdict.index(ii) for ii in jj["ingredients"]] for jj in ingrotot if jj["id"] in list_rec_eff}
		self.catgdictlist={jj["id"]:catgdict.index(jj["category"]) for jj in ingrotot if jj["id"] in list_rec_eff}
		return
		
	def image_labeler(self):
		self.image_labels=
		return

	def ingred_freq(recipe_ids):
		"""
		This method simply computes the frequencies of every class (=ingredient).
		"""
		for rec_id in recipe_ids:
			self.ingred_freq+=self.image_labels[rec_id][1]
		return

	def sieve(self,recipe_ids):
		"""
		This is an algorithm I came up with to balance out my dataset for training. 
		It takes in a list of recipe_ids and returns a smaller list which is more balanced.
		It achieves this by selecting elements randomly, where the probability of being selected depends on the 
		presence or absence of unbalanced ingredients, in a hierarchical manner (i.e. the most unbalanced class 
		present in the data point determines this probabilty, irrespective of whether more balanced ingredients 
		are there or not). Please notice that, given the correlations between ingredients, this is not an easy
		task and it will lead to very unsatisfactory results if one were to have a very large ingredient list 
		(where some ingredient will be extremely rare e.g.).
		"""
		histemp=histofilter(recipe_ids)
		histemp=list(histemp)+[len(recipe_ids)-min(histemp)]
		indsort=sorted(range(len(histemp)), key=lambda k: histemp[k])
		histemp=[int(round(ii/min(histemp))) for ii in histemp]
		numin=len(indsort)-1
		newfiltero=[]
		for ii in images_filtero:
			pop=img_labels[ii][1]
			i=0
			while i<numin and pop[indsort[i]]==0.:
				i+=1
			if random.randint(1,histemp[indsort[i]])==1:
				newfiltero+=[ii]
		return newfiltero