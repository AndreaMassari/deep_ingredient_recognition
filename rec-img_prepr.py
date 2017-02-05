"""
This is the first module of the project, it simply contains the image preprocessing through the VGG16 
conv netw and some loaders. In the future, scraping might be included in this module
"""

i_list_def=["beef","chicken","pork","potatoes","eggs","beans","tomatoes","corn"]

class DIRPreproc:

	def __init__(self,datadirec='private/data/allrecipesscrepo',home='',ingredlist=i_list_def):
		self.datadir=datadirec
		self.scriptdir=home #is this even useful?
		self.list_rec_categ=[]
		self.list_rec_ing=[]
		self.list_rec_effective=[]
		self.ing_list=ingredlist
		self.categ_dict={}
		self.categ_dict_uptodate=False
		self.ingcat_dictlist=[]
		self.ingcat_dict={}
		return

	def categ_loader(self):
		"""
		This method loads recipe category information from the scraped categ.csv file
		"""
		with open(self.datadirec+'/categ.csv') as csvfile:
			reader = csv.reader(csvfile)
			for rec in reader:
				recipe_id=rec[0]
				category=rec[1]
				#add some warning in case the following condition doesn't hold but don't let
				# the warning take too much space
				#if recipe_id not in categ_dict.keys():
				self.list_rec_categ+=[recipe_id]
				self.categ_dict[recipe_id]=category
		categ_dict_uptodate=True
		return

	def ingred_loader(self):
		"""
		This method loads recipe ingredient descriptions from the scraped ingred.csv file
		"""
		if not categ_dict_uptodate:
			print('Run categ_loader() first!!')
			return
		with open(self.datadirec+'/ingred.csv') as csvfile:
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
				if categ_dict[recipe_id]=="Recipes":#this is because some recipes have a generic 'Recipes' 
													#category and I want to instead use their own name as category
					category='**'+recipe_name
				else:
					category=categ_dict[recipe_id]
				ingredients=list(temp_dict_entry)
				temp_dict={"id":recipe_id,"name":recipe_name,"ingredients":ingredients,"category":category}
				self.ingcat_dictlist+=[temp_dict]
				self.ingcat_dict["id"]=[recipe_name,category,ingredients]
		return

	def dictlists_saver(self):
		"""
		This method saves the large dictionaries created by the previous method
		"""
		with open(homedir+'/ingcat_dictlist.json', 'w') as fp:
			json.dump(self.ingcat_dictlist, fp)

		with open(homedir+'/ingcat_dict.json', 'w') as fp:
			json.dump(self.ingcat_dict, fp)
		return

	def img_preproc(self):
		"""
		This method processes images through VGG16 with no top layers and saves output to disc
		"""
		for ii in self.list_rec_eff:
			path_rec=self.datadir+'/'+ii
			path_rec_out=self.datadir+'/preprocessed_img/'+ii
			os.mkdir(path_rec_out)
			for jj in os.listdir(path_rec):
				path_in=path_rec+'/'+jj
				if jj!='.DS_Store':
					img_in=ndimage.imread(path_in)[13:-13,13:-13,:]
					img_out=modelvggc.predict(np.array([img_in]))[0]
					path_out=path_rec_out+'/'+jj
					np.save(path_out,img_out)
		return