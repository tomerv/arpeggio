SUMMARY & USAGE LICENSE
=============================================

This implementation package was written by Yue Shi, multimedia information retrieval lab at Delft University of Technology. 
The detail of the algorithm can be found in our paper, which is mentioned below.
This implementation code is publicly available for usage with any research purposes under the following conditions:
	* Neither Delft University of Technology nor any of the researchers involved can guarantee the correctness of the code, its suitability for any particular purpose, or the validity of results based on the use of the package.
	* The user must acknowledge the use of the implementation package in publications resulting from the use of the package, by citing our paper:
		Shi, Y., Larson, M., and Hanjalic, A., (2010) "List-wise learning to rank with matrix factorization for collaborative filtering". In Proc. RecSys '10, 269-272, ACM.
	* The user may not redistribute the package without separate permission.


DETAILED DESCRIPTIONS OF functions
==============================================

listrank.m

	A main matlab function that learns latent user features and latent item features from training data. Explanations on parameters can be found be "help listrank".
	

logf.m and logfd.m

	Subfunctions that are called by the main function
	
	
Example.mat

	It contains a "Traindata" and a "Testdata". The data is a subset from MovieLens 100K dataset (http://www.grouplens.org/node/73). The "Traindata" can be used as the input for the main function, and the "Testdata" can be used to evaluate the results.
	
	
	
CONTACT
==============================================

For any questions regarding the package, please email to Yue Shi (y.shi@tudelft.nl)
