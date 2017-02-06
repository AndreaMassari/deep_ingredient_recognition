"""
This is the first module of the project, it simply contains the image preprocessing through the VGG16 
conv netw and some loaders. In the future, scraping might be included in this module
"""
import csv
import os
import os.path
import json
import numpy as np
from keras.applications.vgg16 import VGG16
from scipy import ndimage

i_list_def=["beef","chicken","pork","potatoes","eggs","beans","tomatoes","corn"]

class RecPreproc:

	def __init__(self,datadirec='private/allrecipesscrepo',home='',ingredlist=i_list_def):
		"""
		Initializer. _effective lists are a subset of the complete lists based on the fact that there are
		at least one valid picture associated with these recipes id's or ingredients or categories resp.
		"""
		self.datadir=datadirec
		self.scriptdir=home #is this even useful?
		self.list_rec_categ=[]
		self.list_rec_ing=[]
		self.list_rec_effective=[]
		self.ing_list=ingredlist
		self.ing_list_effective=[]
		self.cat_list_effective=[]
		self.categ_dict={}
		self.categ_dict_uptodate=False
		self.ingcat_dictlist=[]
		self.ingcat_dict={}
		return

	def factsandfigs(self):
		"""
		This is just a convenient lookup function to check the overall numbers
		"""
		print("There are",len(self.ingcat_dict), "total recipes in our database")
		print("There are", len(self.list_rec_effective), "distinct recipes with at least one image in our dataset")
		print("There are", len(self.cat_list_effective), "distinct food categories in our picture dataset")
		print("There are", len(self.ing_list_effective), "distinct ingredients in our picture dataset")
		return

	def categ_loader(self):
		"""
		This method loads recipe category information from the scraped categ.csv file
		"""
		with open(self.datadir+'/categ.csv') as csvfile:
			reader = csv.reader(csvfile)
			for rec in reader:
				recipe_id=rec[0]
				category=rec[1]
				#add some warning in case the following condition doesn't hold but don't let
				# the warning take too much output space
				#if recipe_id not in categ_dict.keys():
				self.list_rec_categ+=[recipe_id]
				self.categ_dict[recipe_id]=category
		self.categ_dict_uptodate=True
		return

	def ingred_loader(self):
		"""
		This method loads recipe ingredient descriptions from the scraped ingred.csv file
		"""
		if not self.categ_dict_uptodate:
			print('Run categ_loader() first!!')
			return
		with open(self.datadir+'/ingred.csv') as csvfile:
			reader =  csv.reader(csvfile)
			for rec in reader:
				recipe_id=rec[0]
				recipe_name=rec[1]
				ingred_descriptions=rec[2:]
				self.list_rec_ing+=[recipe_id]
				temp_dict_entry=set([])
				#this is a homemade horrible tokenizer/filter
				for ingred_descr in ingred_descriptions:
					padded_ingred_descr=" "+ingred_descr+" "
					for ingred in self.ing_list:
						for suffix in {" ",".",","}:
							ingred_suff=ingred+suffix
							for prefix in {" ",".",","}:
								if prefix+ingred_suff in padded_ingred_descr: #if, say, " egg." in " One boiled egg. "
									temp_dict_entry.add(ingred)
				category=self.categ_dict[recipe_id]
				if category=="Recipes":#this is because some recipes have a generic 'Recipes' 
													#category and I want to instead use their own name as category
					category='**'+recipe_name
				ingredients=list(temp_dict_entry)
				temp_dict={"id":recipe_id,"name":recipe_name,"ingredients":ingredients,"category":category}
				self.ingcat_dictlist+=[temp_dict]
				self.ingcat_dict[recipe_id]=[recipe_name,category,ingredients]
		return

	def effective_list_maker(self):
		inglisttemp=[]
		catlisttemp=[]
		for recipe in self.ingcat_dictlist:
			#change with something like if os.path.isfile(file_path)
			if len(os.listdir(self.datadir+"/"+recipe['id'])):
				inglisttemp+=recipe["ingredients"]
				catlisttemp+=[recipe["category"]]
				self.list_rec_effective+=[recipe['id']]
		self.ing_list_effective=list(set(inglisttemp))
		self.ing_list_effective.sort()
		self.cat_list_effective=list(set(catlisttemp))
		self.cat_list_effective.sort()
		return


	def dictlists_saver(self,tag=""):
		"""
		This method saves the large dictionaries created by the previous methods. 
		A tag in the filename is optional.
		"""
		with open(self.datadir+'/ingcat_dictlist'+tag+'.json', 'w') as fp:
			json.dump(self.ingcat_dictlist, fp)

		with open(self.datadir+'/ingcat_dict'+tag+'.json', 'w') as fp:
			json.dump(self.ingcat_dict, fp)

		with open(self.datadir+'/ing_list_effective'+tag+'.json', 'w') as fp:
			json.dump(self.ing_list_effective, fp)

		with open(self.datadir+'/cat_list_effective'+tag+'.json', 'w') as fp:
			json.dump(self.cat_list_effective, fp)
			
		with open(self.datadir+'/list_rec_effective'+tag+'.json', 'w') as fp:
			json.dump(self.list_rec_effective, fp)

		return

class ImgPreproc:
	"""
	This class runs images through VGG16 and saves the outputs which will be loaded by the classifier
	"""

	
	def __init__(self,datadirec='private/allrecipesscrepo',img_width=224, img_height=224,load_list_rec_eff=""):
		"""
		Here I store simply recipe ids and the VGG model, initialized
		"""
		self.model = VGG16(weights='imagenet', include_top=False,input_shape=(img_width, img_height,3))
		self.id_ranges=[]
		self.datadir=datadirec
		if str(type(load_list_rec_eff))=='rec_img_preprocessing.DIRPreproc':
			#supply a recipe preprocessing class and it'll inherit its effective list
			self.list_rec_effective=load_list_rec_eff.list_rec_effective
		elif type(load_list_rec_eff)==str:
			#supply a tag and it'll load the appropriate file previously saved by the saver function above
			with open(self.datadir+'/list_rec_eff'+load_list_rec_eff+'.json') as file:
				self.list_rec_effective=json.load(file)
		else:
			self.list_rec_effective=[]
		return

	def make_list_rec_eff(self):
		"""
		Here I'm thinking of supplying the class with a method to go over the ingred directories themself
		and build its list_rec_effective but I recommend loading from previous class output
		"""
		print("This does nothing so far, apologies.")
		return 

	def img_preproc(self,id_range):
		"""
		This function processes images through VGG16 with no top layers and saves output to disc
		"""
		id_list=self.list_rec_effective[id_range[0]:id_range[1]]
		id_list_new=[]
		for recipe_id in id_list:
			path_rec=self.datadir+'/'+recipe_id
			path_rec_out=self.datadir+'/preprocessed_img/'+recipe_id
			if os.path.isdir(path_rec_out):
				print("Folder "+path_rec_out+" exists already. We won't overwrite existing files, just add to them.")
			else:
				os.mkdir(path_rec_out)
			for image_name in os.listdir(path_rec):
				path_img_in=path_rec+'/'+image_name
				path_img_out=path_rec_out+'/'+image_name
				if image_name!='.DS_Store' and not os.path.isfile(path_img_out+'.npy'):
					print("Writing new file at "+path_img_out+'.npy')
					img_in=ndimage.imread(path_img_in)[13:-13,13:-13,:]
					img_out=self.model.predict(np.array([img_in]))[0]
					np.save(path_img_out,img_out)
			id_list_new+=[recipe_id]
		self.id_ranges+=id_list_new
		return



