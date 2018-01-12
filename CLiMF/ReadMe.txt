SUMMARY & USAGE LICENSE
=============================================

This implementation package was written by Yue Shi, multimedia information retrieval lab at Delft University of Technology. 
The detail of the algorithm can be found in our paper, which is mentioned below.
This implementation code is publicly available for usage with any research purposes under the following conditions:
	* Neither Delft University of Technology, Telefonica Research nor any of the researchers involved can guarantee the correctness of the code, its suitability for any particular purpose, or the validity of results based on the use of the package.
	* The user must acknowledge the use of the implementation package in publications resulting from the use of the package, by citing our paper:
		Y. Shi, A. Karatzoglou, L. Baltrunas, M. A. Larson, N. Oliver, and A. Hanjalic, (2012) "CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering". In Proc. RecSys '12, 139-146, ACM.
	* The user may not redistribute the package without separate permission.


DETAILED DESCRIPTIONS OF functions
==============================================

proto_CLiMF_training.m

	A main matlab function that learns latent user factors and latent item factors from training data. Explanations on parameters can be found inside the script.
	

logf.m and logfd.m

	Subfunctions that are called by the main function. Logistic function and its derivative.


mrr_metric.m

	Subfunction called by the main function. MRR measure.

	
	
EP25_UPL5.mat 

	"Traindata" and a "Testdata". The data is a subset from Epinions trust network dataset (http://www.trustlet.org/wiki/Downloaded_Epinions_dataset). 


EP25_UPL5_indrated.mat
	
	Index of rated items in the "Traindata".
	
	
CONTACT
==============================================

For any questions regarding the package, please email to Yue Shi (y.shi@tudelft.nl)
