#include "BagOfWordsRepresentation.h"
#include <exception>
#include <string>
#include <omp.h>


// vanilla string split operation.  Direct copy-paste from stack overflow
// source: http://stackoverflow.com/questions/236129/splitting-a-string-in-c
std::vector<std::string> &BagOfWordsRepresentation::split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> BagOfWordsRepresentation::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}

int BagOfWordsRepresentation::bruteForceMatch(cv::Mat &feature)
{
	int shortest_distance = INT_MAX;
	int shortest_index = -1;

	for (int i = 0; i < clusters_for_matching.size(); i++)
	{
		unsigned int dist = hammingDistance(feature, clusters_for_matching[i]);
		if (dist < shortest_distance)
		{
			shortest_distance = dist;
			shortest_index = i;
		}
	}
	return shortest_index;
}

unsigned int BagOfWordsRepresentation::hammingDistance(cv::Mat &a, cv::Mat &b)
{
	unsigned int distance = 0;
	for (int row = 0; row < a.rows; ++row)
	{
		for (int col = 0; col < a.cols; ++col)
		{
			distance += hammingDistance(a.at<unsigned char>(row, col), b.at<unsigned char>(row, col));
		}
	}

	return distance;
}

unsigned int BagOfWordsRepresentation::hammingDistance(unsigned char a, unsigned char b)
{
	unsigned int hamming_distance = 0;
	// start as 0000 0001
	unsigned int bit = 1;

	// get the xor of a and b, each 1 in the xor adds to the hamming distance...
	unsigned int xor_result = a ^ b;

	// now count the bits, using 'bit' as a mask and performing a bitwise AND
	for (bit = 1; bit != 0; bit <<= 1)
	{
		if ((xor_result & bit) != 0)
		{
			hamming_distance++;
		}
	}

	return hamming_distance;
}

cv::Mat BagOfWordsRepresentation::buildHistogram(std::string &file, bool &success)
{
	success = false;

	cv::Mat histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
	for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		histogram.at<float>(0, col) = 0;

	// open file.
	ifstream input_file(file);
	string line;

	while (std::getline(input_file, line))
	{
        //cout << "line: " << line << endl;
		// discard first 6 values.
		std::istringstream iss(line);
		double discard;

		for (unsigned i = 0; i < 6; ++i)
		{
			iss >> discard;
		}

		// read line and parse into FEATURE_DIMENSIONALITY-dim vector.
		cv::Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_8U);
		float elem;
		
		for (unsigned int i = 0; i < FEATURE_DIMENSIONALITY; ++i)
		{
			iss >> elem;
			feature_vector.at<unsigned char>(0, i) = (unsigned char)elem;
		}

		// match that vector against centroids to assign to correct codeword.
		// brute force match each mofreak point against all clusters to find best match.
		//std::vector<cv::DMatch> matches;
		//bf_matcher->match(feature_vector, matches);
        int best_match = bruteForceMatch(feature_vector);

		// + 1 to that codeword
		//histogram.at<float>(0, matches[0].imgIdx) = histogram.at<float>(0, matches[0].imgIdx) + 1;
		histogram.at<float>(0, best_match) = histogram.at<float>(0, best_match) + 1;
		success = true;
		feature_vector.release();

	}

    /* debug file info
    if (input_file.eof()) { std::cout << "EOFBIT!!!" << endl;}
    if (input_file.bad()) { std::cout << "BADBIT!!!" << endl;}
    if (input_file.fail()) { std::cout << "FAILBIT!!!" << endl;}
    if (!input_file.is_open()) { std::cout << "no está abierto" << endl;}
    if (input_file.is_open()) {
        std::cout << "está abierto, se va a cerrar" << endl;
        input_file.close();
    }
    if (input_file.is_open()) { std::cout << "quedó abierto" << endl;}
    */

	if (!success)
		return histogram;

	// after doing that for all lines in file, normalize.
	float histogram_sum = 0;
	for (int col = 0; col < histogram.cols; ++col)
	{
		histogram_sum += histogram.at<float>(0, col);
	}

	for (int col = 0; col < histogram.cols; ++col)
	{
		histogram.at<float>(0, col) = histogram.at<float>(0, col)/histogram_sum;
	}

	return histogram;
}

void BagOfWordsRepresentation::loadClusters(int group)
{
    //limpio los clusters del grupo anterior
    bf_matcher->clear();
    clusters_for_matching.clear();

    clusters = new cv::Mat(NUMBER_OF_CLUSTERS, FEATURE_DIMENSIONALITY, CV_8U);

    string cluster_path = SVM_PATH + "/clusters" + std::to_string(group+1) + ".txt";

	ifstream cluster_file(cluster_path);
	if ((!cluster_file.is_open()) || cluster_file.bad())
	{
		cout << "could not open clusters file" << endl;
		exit(1);
	}

	string line;
	unsigned int row = 0;
	
	while (std::getline(cluster_file, line))
	{
		cv::Mat single_cluster(1, FEATURE_DIMENSIONALITY, CV_8U);
		float elem;
		for (unsigned int col = 0; col < FEATURE_DIMENSIONALITY; ++col)
		{
			cluster_file >> elem;
			clusters->at<unsigned char>(row, col) = (unsigned char)elem;
			single_cluster.at<unsigned char>(0, col) = (unsigned char)elem;
		}
		clusters_for_matching.push_back(single_cluster);
		++row;
	}
	cluster_file.close();

	// add clusters to bruteforce matcher.
	bf_matcher->add(clusters_for_matching);
}

// when doing this, make sure all mofreak points for the video are in ONE file, to avoid missing cuts.
void BagOfWordsRepresentation::computeSlidingBagOfWords(std::string &file, int alpha, int label, ofstream &out)
{

	bool over_alpha_frames = false;
	string distance_file = file;

	std::list<cv::Mat> histograms_per_frame;
	std::vector<cv::Mat> feature_list; // for just the current frame.

	// get start frame and start loading mofreak features...
	ifstream input_file(file);
	string line;
	int current_frame;

	std::getline(input_file, line);
	std::istringstream init(line);
	double discard;
	
	// ignore the first two values. the 3rd is the frame.
	init >> discard;
	init >> discard;
	init >> current_frame;

	// 4 5 and 6 aren't relevant either.
	init >> discard;
	init >> discard;
	init >> discard;

	// load the first feature to the list.  Now we're rolling.
	cv::Mat ftr_vector(1, FEATURE_DIMENSIONALITY, CV_32FC1);
	float elem;

	for (unsigned int i = 0; i < FEATURE_DIMENSIONALITY; ++i)
	{
		init >> elem;
		ftr_vector.at<float>(0, i) = elem;
	}
	feature_list.push_back(ftr_vector);

	// now the first line is out of the way.  Load them all
	// ---------------------------------------------------

	while (std::getline(input_file, line))
	{
		std::istringstream iss(line);
		// first two still aren't relevant... x,y position.
		iss >> discard;
		iss >> discard;

		int frame;
		iss >> frame;
		// next 3 aren't relevant.
		iss >> discard;
		iss >> discard;
		iss >> discard;

		// still on the same frame.  Just add to our feature list.
		if (frame == current_frame)
		{
			cv::Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_32FC1);
			for (unsigned i = 0; i < FEATURE_DIMENSIONALITY; ++i)
			{
				iss >> elem;
				feature_vector.at<float>(0, i) = elem;
			}
			
			feature_list.push_back(feature_vector);
		}

		// new frame.  need to compute the hist on that frame and possibly compute a new BOW feature.
		else
		{
			// 1: compute the histogram...
			cv::Mat new_histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
			for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
				new_histogram.at<float>(0, col) = 0;

			for (auto it = feature_list.begin(); it != feature_list.end(); ++it)
			{
				// match that vector against centroids to assign to correct codeword.
				// brute force match each mofreak point against all clusters to find best match.
				std::vector<cv::DMatch> matches;
				bf_matcher->match(*it, matches);	

				// + 1 to that codeword
				new_histogram.at<float>(0, matches[0].imgIdx) = new_histogram.at<float>(0, matches[0].imgIdx) + 1;
			}

			// 2: add the histogram to our list.
			histograms_per_frame.push_back(new_histogram);

			// histogram list is at capacity!
			// compute summed histogram over all histograms as BOW feature.
			// then pop.
			// finally, write this new libsvm-worthy feature to file
			if (histograms_per_frame.size() == alpha)
			{
				over_alpha_frames = true;

				cv::Mat histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
				for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
					histogram.at<float>(0, col) = 0;

				// sum over the histograms we have.
				for (auto it = histograms_per_frame.begin(); it != histograms_per_frame.end(); ++it)
				{
					for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
					{
						histogram.at<float>(0, col) += it->at<float>(0, col);
					}
				}

				// remove oldest histogram.
				histograms_per_frame.pop_front();

				// normalize the histogram.
				float normalizer = 0.0;
				for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
				{
					normalizer += histogram.at<float>(0, col);
				}

				for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
				{
					histogram.at<float>(0, col) = histogram.at<float>(0, col) / normalizer;
				}

				// write to libsvm...
				stringstream ss;
				string current_line;

				ss << label << " ";
				for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
				{
					ss << (col + 1) << ":" << histogram.at<float>(0, col) << " ";
				}
				current_line = ss.str();
				ss.str("");
				ss.clear();

				out << current_line << endl;
			}

			// reset the feature list for the new frame.
			feature_list.clear();

			// add current line to the _new_ feature list.
			cv::Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_32FC1);
			for (int i = 0; i < FEATURE_DIMENSIONALITY; ++i)
			{
				iss >> elem;
				feature_vector.at<float>(0, i) = elem;
			}
			
			feature_list.push_back(feature_vector);
			current_frame = frame;
		}
	}

	// if we didn't get to write for this file, write it here.
	if (!over_alpha_frames)
	{
		cv::Mat histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
		for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
			histogram.at<float>(0, col) = 0;

		// sum over the histograms we have.
		for (auto it = histograms_per_frame.begin(); it != histograms_per_frame.end(); ++it)
		{
			for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
			{
				histogram.at<float>(0, col) += it->at<float>(0, col);
			}
		}

		// normalize the histogram.
		float normalizer = 0.0;
		for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		{
			normalizer += histogram.at<float>(0, col);
		}

		for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		{
			histogram.at<float>(0, col) = histogram.at<float>(0, col) / normalizer;
		}

		// write to libsvm...
		stringstream ss;
		string current_line;

		ss << label << " ";
		for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		{
			ss << (col + 1) << ":" << histogram.at<float>(0, col) << " ";
		}
		current_line = ss.str();
		ss.str("");
		ss.clear();

		out << current_line << endl;
	}
}

void BagOfWordsRepresentation::extractMetadata(std::string filename, int &action, int &group, int &clip_number)
{
    if (dataset == UCF101 || dataset == KTH)
	{
		std::vector<std::string> filename_parts = split(filename, '_');

		// extract action.
		std::string parsed_action = filename_parts[1];

        if (dataset == KTH)
        {
            // get the action.
            if (boost::contains(parsed_action, "box"))
            {
                action = 1;
            }
            else if (boost::contains(parsed_action, "clap"))
            {
                action = 2;
            }
            else if (boost::contains(parsed_action, "wav"))
            {
                action = 3;
            }
            else if (boost::contains(parsed_action, "jog"))
            {
                action = 4;
            }
            else if (boost::contains(parsed_action, "run"))
            {
                action = 5;
            }
            else if (boost::contains(parsed_action, "walk"))
            {
                action = 6;
            }
            else
            {
                std::cout << "Didn't find action: " << parsed_action << std::endl;
                exit(1);
            }
        }

		// extract group
		if (dataset == KTH)
		{
			std::stringstream(filename_parts[0].substr(filename_parts[0].length() - 2, 2)) >> group;
		}
		else
		{
			std::string parsed_group = filename_parts[2].substr(1, 2);
			std::stringstream(parsed_group) >> group;
		}
		group--; // group indices start at 0, not 1, so decrement.

		// extract clip number.
		std::string parsed_clip = filename_parts[3].substr(2, 2);
		std::stringstream(parsed_clip) >> clip_number;
	}
    else if (dataset == RUGBY) {


        /* vieja denominacion de videos
        std::vector<std::string> filename_parts = split(filename, '_');

        // extract action.
        std::string parsed_action = filename_parts[1];

        // get the action.
        if (boost::contains(parsed_action, "line"))
        {
            action = 1;
        }
        else if (boost::contains(parsed_action, "scrum"))
        {
            action = 2;
        }
        else if (boost::contains(parsed_action, "juego"))
        {
            action = 3;
        }
        else
        {
            std::cout << "Didn't find action: " << parsed_action << std::endl;
            exit(1);
        }

        // extract group
        std::stringstream(filename_parts[0]) >> group;
        group--; // group indices start at 0, not 1, so decrement.

        // extract clip number.
        std::string parsed_clip = filename_parts[2]; //filename_parts[3].substr(2, 2);
        std::stringstream(parsed_clip) >> clip_number;
        */

        std::vector<std::string> filename_parts = split(filename, '_');

        // extract action.
        std::string parsed_action = filename_parts[0];

        // get the action.
        if (boost::contains(parsed_action, "line"))
        {
            action = 1;
        }
        else if (boost::contains(parsed_action, "scrum"))
        {
            action = 2;
        }
        else if (boost::contains(parsed_action, "juego"))
        {
            action = 3;
        }
        else
        {
            std::cout << "Didn't find action: " << parsed_action << std::endl;
            exit(1);
        }

        // extract group
        std::stringstream(filename_parts[1].substr(5, 4)) >> group;
        group--; // group indices start at 0, not 1, so decrement.

        // extract clip number.
        std::string parsed_clip = filename_parts[2].substr(4, 4);
        std::stringstream(parsed_clip) >> clip_number;

    }
}

void BagOfWordsRepresentation::intializeBOWMemory(string SVM_PATH)
{

	// initialize the output files and memory for BOW features for each grouping.
	for (int group = 0; group < NUMBER_OF_GROUPS; ++group)
	{
		stringstream training_filepath, testing_filepath;
		training_filepath << SVM_PATH << "/" << group + 1 << ".train";
		testing_filepath << SVM_PATH << "/" << group + 1 << ".test";

		ofstream *training_filestream = new ofstream(training_filepath.str());
		ofstream *testing_filestream = new ofstream(testing_filepath.str());

		training_files.push_back(training_filestream);
		testing_files.push_back(testing_filestream);

		training_filepath.str("");
		training_filepath.clear();
		testing_filepath.str("");
		testing_filepath.clear();

		std::vector<string> training_features;
		std::vector<string> testing_features;
		bow_training_crossvalidation_sets.push_back(training_features);
		bow_testing_crossvalidation_sets.push_back(testing_features);
	}
}

void BagOfWordsRepresentation::convertFileToBOWFeature(int group, std::string file, std::vector<cv::Mat> bow_features, int i)
{
	boost::filesystem::path file_path(file);
	boost::filesystem::path file_name = file_path.filename();
	std::string file_name_str = file_name.generic_string();

    std::cout << "thread: " << omp_get_thread_num() << " file: " << file << endl;

    // extract the metadata from this file, such as the group and action performed.
    int action, video_group, video_number;
    extractMetadata(file_name_str, action, group, video_number);

    //for (int g = 1; g <= NUMBER_OF_GROUPS; g++)
    //{

        /*
        Now, extract each mofreak features and assign them to the correct codeword.
        buildHistogram returns a histogram representation (1 row, num_clust cols)
        of the bag-of-words feature.  If for any reason the process fails,
        the "success" boolean will be returned as false
        */
        bool success;
        cv::Mat bow_feature;

        //usa el cluster que corresponde al grupo
        //al llamar a loadClusters limpia clusters_for_matching antes
        //de cargar los del grupo correspondiente
        try
        {
            bow_feature = buildHistogram(file, success);
            bow_features[i] = bow_feature;
        }
        catch (cv::Exception &e)
        {
            cout << "Error: " << e.what() << endl;
            exit(1);
        }

        if (!success)
        {
            std::cout << "Bag-of-words feature construction was unsuccessful.  Investigate." << std::endl;
            exit(1);
        //	continue;
        }

        /*
        Prepare each histogram to be written as a line to multiple files.
        It gets assigned to each training file, except for the training
        file where the group id matches that leave-one-out iteration

        stringstream ss;
        ss << action << " ";

        for (int col = 0; col < bow_feature.cols; ++col)
        {
            ss << (int)(col + 1) << ":" << (float)bow_feature.at<float>(0, col) << " ";
        }

        string current_line;
        current_line = ss.str();
        ss.str("");
        ss.clear();



        for (int g = 0; g < NUMBER_OF_GROUPS; ++g)
        {

            if (g == group)
            {

                bow_testing_crossvalidation_sets[g].push_back(current_line);
            }
            else
            {
                bow_training_crossvalidation_sets[g].push_back(current_line);
            }
         }
         */
    //}
}

void BagOfWordsRepresentation::writeBOWFeaturesToVectors(int group, std::vector<std::string> mofreak_files, std::vector<cv::Mat> bow_features)
{
    stringstream ss;
    int action, video_group, video_number;

    std::cout << "thread: " << omp_get_thread_num() << " cargar en vector" << endl;

    for (int i = 0; i < mofreak_files.size(); i++)
    {
        string filename = mofreak_files[i].substr(mofreak_files[i].find_last_of("/")+1);
        extractMetadata(filename, action, video_group, video_number);
        ss << action << " ";

        cv::Mat feat = bow_features[i];
        if (feat.empty())
        {
            continue;
        }
        else
        {

            for (int col = 0; col < feat.cols; ++col)
            {
                ss << (int)(col + 1) << ":" << (float)feat.at<float>(0, col) << " ";
            }

            string current_line;
            current_line = ss.str();
            ss.str("");
            ss.clear();

//            for (int g = 0; g < NUMBER_OF_GROUPS; ++g)
//            {
//                if (g == group)
//                    bow_testing_crossvalidation_sets[g].push_back(current_line);
//                else
//                    bow_training_crossvalidation_sets[g].push_back(current_line);
//            }
            if (video_group == group)
                bow_testing_crossvalidation_sets[group].push_back(current_line);
            else
                bow_training_crossvalidation_sets[group].push_back(current_line);
        }
    }
}

void BagOfWordsRepresentation::writeBOWFeaturesToFiles()
{
	// ensure that we have the correct number of open files
	if (bow_training_crossvalidation_sets.size() != NUMBER_OF_GROUPS || bow_testing_crossvalidation_sets.size() != NUMBER_OF_GROUPS)
	{
		cout << "Incorrect number of training or testing file lines.  Check mapping from bow feature to test/train files" << endl;
		exit(1);
	}

	// for each group, write the training and testing cross-validation files.
	for (int i = 0; i < NUMBER_OF_GROUPS; ++i)
	{
		cout << "number of training features: " << bow_training_crossvalidation_sets[i].size() << endl;
		for (unsigned line = 0; line < bow_training_crossvalidation_sets[i].size(); ++line)
		{
			try
			{
				*training_files[i] << bow_training_crossvalidation_sets[i][line] << endl;
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				exit(1);
			}
		}
		
		cout << "number of testing features: " << bow_testing_crossvalidation_sets[i].size() << endl;
		for (unsigned line = 0; line < bow_testing_crossvalidation_sets[i].size(); ++line)
		{
			try
			{
				*testing_files[i] << bow_testing_crossvalidation_sets[i][line] << endl;
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				exit(1);
			}
		}
	}

	cout << "Finished writing to cross-validation files." << endl;

	// close the libsvm training and testing files.
	for (int i = 0; i < NUMBER_OF_GROUPS; ++i)
	{
		training_files[i]->close();
		testing_files[i]->close();

		delete training_files[i];
		delete testing_files[i];
	}

	cout << "Closed all cross-validation files. " << endl;

    training_files.clear();
    testing_files.clear();
}

//no se usa?
//void BagOfWordsRepresentation::computeBagOfWords(string SVM_PATH, string MOFREAK_PATH, string METADATA_PATH)
//{
//	if (dataset == KTH || dataset == UCF101)
//	{
//		for (int i = 0; i < NUMBER_OF_GROUPS; ++i)
//		{
//			stringstream training_string;
//			stringstream testing_string;

//			training_string << SVM_PATH + "/left_out_" << i + 1 << ".train";
//			testing_string << SVM_PATH + "/left_out_" << i + 1 << ".test";

//			ofstream *training_file = new ofstream(training_string.str());
//			ofstream *testing_file = new ofstream(testing_string.str());

//			training_string.str("");
//			training_string.clear();
//			testing_string.str("");
//			testing_string.clear();

//			training_files.push_back(training_file);
//			testing_files.push_back(testing_file);

//			vector<string> training_lines;
//			vector<string> testing_lines;
//			bow_training_crossvalidation_sets.push_back(training_lines);
//			bow_testing_crossvalidation_sets.push_back(testing_lines);
//		}
//	}
	
//	cout << "parallelizing bag-of-words matches across files." << endl;
//#pragma omp parallel for
//	for (int i = 0; i < files.size(); ++i)
//	{
//		convertFileToBOWFeature(files[i]);
//	}
//	cout << "done bag-of-words feature construction" << endl;

//	/*
//	Finally, after all of the BOW features have been computed,
//	we write them to the corresponding files.
//	This is outside of the parallelized loop,
//	since writing to a file isn't thread-safe.
//	*/
//	writeBOWFeaturesToFiles();
//}

BagOfWordsRepresentation::BagOfWordsRepresentation(int num_clust, int ftr_dim, std::string svm_path, int num_groups, int dset) :
    NUMBER_OF_CLUSTERS(num_clust), FEATURE_DIMENSIONALITY(ftr_dim), SVM_PATH(svm_path), NUMBER_OF_GROUPS(num_groups), dataset(dset)
{
    bf_matcher = new cv::BFMatcher(cv::NORM_HAMMING);
    //loadClusters();

    motion_descriptor_size = 8;
    appearance_descriptor_size = 8;
    motion_is_binary = true;
    appearance_is_binary = true;
}

BagOfWordsRepresentation::BagOfWordsRepresentation(std::vector<std::string> &file_list, 
	int num_clust, int ftr_dim, int num_groups, bool appearance_is_bin, 
	bool motion_is_bin, int dset, std::string svm_path) :

	NUMBER_OF_CLUSTERS(num_clust), FEATURE_DIMENSIONALITY(ftr_dim), 
	NUMBER_OF_GROUPS(num_groups), motion_is_binary(motion_is_bin), 
	appearance_is_binary(appearance_is_bin), dataset(dset),
	SVM_PATH(svm_path)
{
	files = file_list;
	bf_matcher = new cv::BFMatcher(cv::NORM_HAMMING);
    //loadClusters();

	// default values.
	motion_descriptor_size = 8; 
	appearance_descriptor_size = 0;
	motion_is_binary = true;
	appearance_is_binary = true;
}

void BagOfWordsRepresentation::setMotionDescriptor(unsigned int size, bool binary)
{
	motion_is_binary = binary;
	motion_descriptor_size = size;
}

void BagOfWordsRepresentation::setAppearanceDescriptor(unsigned int size, bool binary)
{
	appearance_is_binary = binary;
	appearance_descriptor_size = size;
}

BagOfWordsRepresentation::~BagOfWordsRepresentation()
{
	clusters->release();
	delete clusters;


}
