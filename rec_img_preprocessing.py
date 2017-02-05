"""
This is the first module of the project, it simply contains the image preprocessing through the VGG16 
conv netw and some loaders. In the future, scraping might be included in this module
"""
import csv
import os
import json

i_list_def=["beef","chicken","pork","potatoes","eggs","beans","tomatoes","corn"]

class DIRPreproc:

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


def img_preproc(id_range,datadir,img_width=224, img_height=224):#typically list_rec_effective[12000:13000]
	"""
	This function processes images through VGG16 with no top layers and saves output to disc
	"""
	modelvggc = VGG16(weights='imagenet', include_top=False,input_shape=(img_width, img_height,3))

	for recipe_id in id_range:
		path_rec=datadir+'/'+recipe_id
		path_rec_out=datadir+'/preprocessed_img/'+recipe_id
		os.mkdir(path_rec_out)
		for image_name in os.listdir(path_rec):
			path_img_in=path_rec+'/'+image_name
			if image_name!='.DS_Store':
				img_in=ndimage.imread(path_img_in)[13:-13,13:-13,:]
				img_out=modelvggc.predict(np.array([img_in]))[0]
				path_img_out=path_rec_out+'/'+image_name
				np.save(path_img_out,img_out)
	return