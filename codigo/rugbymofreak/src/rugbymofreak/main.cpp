#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <exception>

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <opencv2/opencv.hpp>

#include "MoFREAKUtilities.h"
#include "Clustering.h"
#include "BagOfWordsRepresentation.h"
#include "SVMInterface.h"

// for debugging the heap (memory leaks, etc)
//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <iomanip>
//#include <crtdbg.h>

#include <omp.h>

using namespace std;
using namespace boost::filesystem;

bool DISTRIBUTED = false;

string MOSIFT_DIR, MOFREAK_PATH, VIDEO_PATH, SVM_PATH, METADATA_PATH; // for file structure
string MOFREAK_NEG_PATH, MOFREAK_POS_PATH; // these are TRECVID exclusive

unsigned int NUM_MOTION_BYTES = 8;
unsigned int NUM_APPEARANCE_BYTES = 8;
unsigned int FEATURE_DIMENSIONALITY = NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES;
unsigned int NUM_CLUSTERS, NUMBER_OF_GROUPS, NUM_CLASSES, ALPHA;

vector<int> possible_classes;
std::deque<MoFREAKFeature> mofreak_ftrs;

enum states {DETECT_MOFREAK, DETECTION_TO_CLASSIFICATION, // standard recognition states
	PICK_CLUSTERS, COMPUTE_BOW_HISTOGRAMS, DETECT, TRAIN, GET_SVM_RESPONSES,}; // these states are exclusive to TRECVID

enum datasets {RUGBY, KTH};

int dataset = RUGBY; //KTH; //UCF101; //HMDB51;
int state = DETECTION_TO_CLASSIFICATION; //DETECT_MOFREAK;

MoFREAKUtilities *mofreak;
SVMInterface svm_interface;

struct Detection
{
	int start_frame;
	int end_frame;
	float score;
	string video_name;

	// override the < operator for sorting.
	bool operator < (const Detection &det) const
	{
		return (score < det.score);
	};
};

// set up some hard-coded parameters that are specific to individual datasets.
// these parameters include things like the number of classes and input/output locations.
void setParameters()
{
    // RUGBY
    if (dataset == RUGBY)
    {
        NUM_CLUSTERS = 1000; //probar
        NUM_CLASSES = 3; //line, scrum, juego
        NUMBER_OF_GROUPS = 8; //no tenemos groups

        for (unsigned i = 0; i < NUM_CLASSES; ++i)
        {
            possible_classes.push_back(i);
        }

        // structural folder info.
        MOSIFT_DIR = "/home/lucia/Documentos/data/mosift/";
        MOFREAK_PATH = "/home/lucia/Documentos/data/mofreak/";
        VIDEO_PATH = "/home/lucia/Documentos/data/videos/";
        SVM_PATH = "/home/lucia/Documentos/data/svm/";
        METADATA_PATH = "";
    }

    // KTH
	else if (dataset == KTH)
	{
        NUM_CLUSTERS = 10;
		NUM_CLASSES = 6;
		NUMBER_OF_GROUPS = 25;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info.
		MOSIFT_DIR = "/home/lucia/data/kth/mosift/";
		MOFREAK_PATH = "/home/lucia/data/kth/mofreak/"; 
		VIDEO_PATH = "/home/lucia/data/kth/videos/";
		SVM_PATH = "/home/lucia/data/kth/svm/";
		METADATA_PATH = "";
	}

}

// cluster MoFREAK points to select codewords for a bag-of-words representation.
void cluster()
{
	cout << "Gathering MoFREAK Features..." << endl;
	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, 0, NUM_CLASSES, possible_classes, SVM_PATH);
	clustering.setAppearanceDescriptor(NUM_APPEARANCE_BYTES, true);
	clustering.setMotionDescriptor(NUM_MOTION_BYTES, true);

    //for each group (partido)
    for (int group = 1; group <= NUMBER_OF_GROUPS; group++)
    {
        clustering.center_row = 0;
        // for each class
        directory_iterator end_iter;
        for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
        {

            if (is_directory(dir_iter->status()))
            {
                // gather all of the mofreak files.
                string mofreak_action = dir_iter->path().filename().generic_string();
                string action_mofreak_path = MOFREAK_PATH + "/" + mofreak_action;
                mofreak->setCurrentAction(mofreak_action);
                std::cout << "action: " << mofreak_action << std::endl;

                // count the number of mofreak files in this class.
                // that way, we can group them for clustering, to avoid memory issues.
                unsigned int file_count = 0;
                for (directory_iterator file_counter(action_mofreak_path);
                    file_counter != end_iter; ++file_counter)
                {
                    if (is_regular_file(file_counter->status())) {
                        std::string path = file_counter->path().string();
                        std::string path_group = path.substr(path.find_last_of('//') + 1, 1);
                        if (group != atoi(path_group.c_str()))
                            file_count++;
                    }
                }

                // maximum number of features to read from each file,
                // to avoid reading in too many mofreak features.
                unsigned int features_per_file = 5000/file_count; //original 50000

                for (directory_iterator mofreak_iter(action_mofreak_path);
                    mofreak_iter != end_iter; ++mofreak_iter)
                {
                    // load each mofreak file's datamofreak_iter
                    if (is_regular_file(mofreak_iter->status()))
                    {
                        std::string path = mofreak_iter->path().string();
                        std::string path_group = path.substr(path.find_last_of('//') + 1, 1);
                        if (group != atoi(path_group.c_str()))
                            mofreak->readMoFREAKFeatures(path, features_per_file);
                        //mofreak->readMoFREAKFeatures(mofreak_iter->path().string(), features_per_file);
                    }
                }

                // the mofreak features are loaded for this class
                // and now, we select clusters.
                cout << "Building data." << endl;
                clustering.buildDataFromMoFREAK(mofreak->getMoFREAKFeatures(), false, false, false, 1); //agreagdo el último false y el 1
                clustering.randomClusters(true);
                mofreak->clearFeatures();
            }
        }
        clustering.writeClusters(group, true);
    }
}


// Convert a file path (pointing to a mofreak file) into a bag-of-words feature.
//void convertFileToBOWFeature(BagOfWordsRepresentation &bow_rep, directory_iterator file_iter)
//{
//	std::string mofreak_filename = file_iter->path().filename().generic_string();
//	if (mofreak_filename.substr(mofreak_filename.length() - 7, 7) == "mofreak")
//	{
//		bow_rep.convertFileToBOWFeature(file_iter->path().string());
//	}
//}

void computeBOWRepresentation()
{
	// initialize BOW representation
    BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES, SVM_PATH, NUMBER_OF_GROUPS, dataset);
    bow_rep.intializeBOWMemory(SVM_PATH);

	// load mofreak files
	std::cout << "Gathering MoFREAK files from " << MOFREAK_PATH << std::endl;
    std::vector<std::string> mofreak_files;
	directory_iterator end_iter;
    for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
    {
        //una carpeta por clase
        if (is_directory(dir_iter->status()))
        {
            std::string action = dir_iter->path().filename().generic_string();
            std::string action_mofreak_path = MOFREAK_PATH + "/" + action;
            std::cout << "action: " << action << std::endl;

            //carga los path de todos los mofreak de una clase
            for (directory_iterator mofreak_iter(action_mofreak_path); mofreak_iter != end_iter; ++mofreak_iter)
            {
                if (is_regular_file(mofreak_iter->status()))
                {
                    {
                        std::string mofreak_filename = mofreak_iter->path().string();
                        if (mofreak_filename.substr(mofreak_filename.length() - 7, 7) == "mofreak")
                        {
                            mofreak_files.push_back(mofreak_filename);
                        }

                        //convertFileToBOWFeature(bow_rep, mofreak_iter);
                    }
                }
            }


        }
        //todas las clases juntas
        else if (is_regular_file(dir_iter->status()))
        {
           //std::cout << "num_threads = " << omp_get_num_threads() << endl;
           //std::cout << "thread_num = " << omp_get_thread_num() << endl;
           //bow_rep.convertFileToBOWFeature(bow_rep, dir_iter);
        }
    }

    for (int g = 0; g < NUMBER_OF_GROUPS; g++)
    {
        std::cout << "Computing histograms for group " << g+1 << std::endl;
        bow_rep.loadClusters(g);
        std::vector<cv::Mat> bow_features = std::vector<cv::Mat>();
        for (int i = 0; i < mofreak_files.size(); i++)
        {
            cv::Mat mat(1, NUM_CLUSTERS, CV_32FC1);
            bow_features.push_back(mat);
        }

        #pragma omp num_thread(4)
        #pragma omp parallel for
        for (int i = 0; i < mofreak_files.size(); i++)
        {
            bow_rep.convertFileToBOWFeature(g, mofreak_files[i], bow_features, i);
        }
        bow_rep.writeBOWFeaturesToVectors(g, mofreak_files, bow_features);


    }
    bow_rep.writeBOWFeaturesToFiles();
    mofreak_files.clear();
	std::cout << "Completed printing bag-of-words representation to files" << std::endl;
}

double classify()
{
	cout << "in eval" << endl;
	// gather testing and training files...
	vector<std::string> testing_files;
	vector<std::string> training_files;
	cout << "Eval SVM..." << endl;
	directory_iterator end_iter;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 5, 5) == "train")
			{
				training_files.push_back(current_file.string());
			}
			else if (filename.substr(filename.length() - 4, 4) == "test")
			{
				testing_files.push_back(current_file.string());
			}
		}
	}

	// evaluate the SVM with leave-one-out.
	std::string results_file = SVM_PATH;
	results_file.append("/svm_results.txt");
	ofstream output_file(results_file);

	string model_file_name = SVM_PATH;
	model_file_name.append("/model.svm");

	string svm_out = SVM_PATH;
	svm_out.append("/responses.txt");

	// confusion matrix.
    cout << "confusion matrix NUM_CLASSES " << NUM_CLASSES << endl;
	cv::Mat confusion_matrix = cv::Mat::zeros(NUM_CLASSES, NUM_CLASSES, CV_32F);
    for (int row = 0; row < confusion_matrix.rows; ++row)
    {
        for (int col = 0; col < confusion_matrix.cols; ++col)
        {
            confusion_matrix.at<float>(row, col) = 0;
        }
    }

	double summed_accuracy = 0.0;
	for (unsigned i = 0; i < training_files.size(); ++i)
	{
		cout << "New loop iteration" << endl;
		SVMInterface svm_guy;
		// tell GUI where we're at in the l-o-o process
		cout << "Cross validation set " << i + 1 << endl;

		// build model.
		string training_file = training_files[i];
		svm_guy.trainModel(training_file, model_file_name);

		// get accuracy.
		string test_filename = testing_files[i];
		double accuracy = svm_guy.testModel(test_filename, model_file_name, svm_out);
		summed_accuracy += accuracy;

		// update confusion matrix.
		// get svm responses.
		vector<int> responses;

		ifstream response_file(svm_out);
		string line;
		while (std::getline(response_file, line))
		{
			int response;
			istringstream(line) >> response;
			responses.push_back(response);
		}
		response_file.close();

		// now get expected output.
		vector<int> ground_truth;

		ifstream truth_file(test_filename);
		while (std::getline(truth_file, line))
		{
			int truth;
			int first_space = line.find(" ");
			if (first_space != string::npos)
			{
				istringstream (line.substr(0, first_space)) >> truth;
				ground_truth.push_back(truth);
			}
		}

		// now add that info to the confusion matrix.
		// row = ground truth, col = predicted..
		for (unsigned int response = 0; response < responses.size(); ++response)
		{
            int row = ground_truth[response] - 1;
            int col = responses[response] - 1;

            confusion_matrix.at<float>(row, col) += 1;
		}
		
		// debugging...print to testing file.
		output_file << training_files[i] <<", " << testing_files[i] << ", " << accuracy << std::endl;
	}	

	// normalize each row.
	// NUM_CLASSES rows/cols (1 per action)
	for (unsigned int row = 0; row < NUM_CLASSES; ++row)
	{
		float normalizer = 0.0;
		for (unsigned int col = 0; col < NUM_CLASSES; ++col)
		{
			normalizer += confusion_matrix.at<float>(row, col);
		}

		for (unsigned int col = 0; col < NUM_CLASSES; ++col)
		{
			confusion_matrix.at<float>(row, col) /= normalizer;
		}
	}

	cout << "Confusion matrix" << endl << "---------------------" << endl;
	for (int row = 0; row < confusion_matrix.rows; ++row)
	{
		for (int col = 0; col < confusion_matrix.cols; ++col)
		{
            cout << fixed << setw(8) << std::setprecision(2) << setfill(' ') <<confusion_matrix.at<float>(row, col) * 100 << ", ";
		}
		cout << endl << endl;
	}

	output_file.close();

	// output average accuracy.
	double denominator = (double)training_files.size();
	double average_accuracy = summed_accuracy/denominator;

	cout << "Averaged accuracy: " << average_accuracy << "%" << endl;

	/*
	write accuracy to file.  
	temporary for testing.
	*/

    ofstream acc_file;
	acc_file.open("accuracy.txt");
	
	acc_file << average_accuracy;
    acc_file.close();

	return average_accuracy;
}



// so, this function will give us sliding window BOW features.
// We can also use this to get our SVM responses to mean-shift away.
// ***********
// Exclusively used for the TRECVID scenario now,
// any remaining examples are deprecated. [TODO]
void computeBOWHistograms(bool positive_examples)
{
	// gather all files int vector<string> mofreak_files
	cout << "Gathering MoFREAK Features..." << endl;
	vector<std::string> mofreak_files;
	directory_iterator end_iter;

	if (DISTRIBUTED)
	{
		MOFREAK_PATH = "mosift/";
	}

	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				mofreak_files.push_back(current_file.string());
			}
		}
	}
	cout << "MoFREAK features gathered." << endl;

	// load clusters.
	BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, FEATURE_DIMENSIONALITY, SVM_PATH, NUMBER_OF_GROUPS, dataset);

	// for each file....
	// slide window of length alpha and use those pts to create a BOW feature.
#pragma omp parallel for
	for (int i = 0; i < mofreak_files.size(); ++i)
	{
		cout << "Computing on " << mofreak_files[i] << endl;
		std::string bow_file = mofreak_files[i];
		bow_file.append(".test");
		ofstream bow_out(bow_file);

		int label = positive_examples ? 1 : -1;
		bow_rep.computeSlidingBagOfWords(mofreak_files[i], ALPHA, label, bow_out);
		bow_out.close();
		cout << "Done " << mofreak_files[i] << endl;
	}
}

void detectEvents()
{
	vector<std::string> response_files;
	directory_iterator end_iter;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.length() > 9)
			{
				if (filename.substr(filename.length() - 13, 13) == "responses.txt")
				{
					response_files.push_back(current_file.string());
				}
			}
		}
	}

	for (auto it = response_files.begin(); it != response_files.end(); ++it)
	{
		cout << "filename: " << *it << endl;
		// read in libsvm output.
		ifstream svm_in(*it);

		// store each value in a list that we can reference.
		vector<float> svm_responses;
		while (!svm_in.eof())
		{
			float response;
			svm_in >> response;
			svm_responses.push_back(response);
		}

		cout << svm_responses.size() << " total SVM responses." << endl;


		// get peaks. [val(x) > val(x - 1) & val(x) > val(x + 1)]
		vector<int> peak_indices;
		for (unsigned int i = 1; i < svm_responses.size() - 1; ++i)
		{
			float response_x = svm_responses[i];
			float response_x_minus_1 = svm_responses[i - 1];
			float response_x_plus_1 = svm_responses[i + 1];

			if ((response_x > response_x_minus_1) && (response_x > response_x_plus_1))
			{
				peak_indices.push_back(i);
			}
		}

		cout << peak_indices.size() << " total detected peaks" << endl;

		// For each of those peaks, run the meanshift-like process to determine if its a window-wise local maxima in the response space.
		// that is, check the alpha/2 points before it and alpha/2 points after it.  If it is the largest value in that window,
		// then this is a candidate detection.
		vector<int> candidate_indices;
		for (auto peak = peak_indices.begin(); peak != peak_indices.end(); ++peak)
		{
			double value = svm_responses[*peak];
			int start_index = max((*peak) - (int)ALPHA, 0);
			int end_index = min((*peak) + (int)ALPHA, (int)svm_responses.size() - 1);
			bool is_local_max = true;

			for (int i = start_index; i < end_index; ++i)
			{
				if (svm_responses[i] > value)
				{
					is_local_max = false;
					break;
				}
			}

			if (is_local_max)
			{
				candidate_indices.push_back(*peak);
			}
		}

		cout << candidate_indices.size() << " detected candidates" << endl;


		// finally, if the detection's response is above our defined threshold, it's a real detection.
		float THRESHOLD = 0;
		unsigned int MAX_DETECTIONS = 30;
		unsigned int MIN_DETECTIONS = 1;
		float STEP = 0.05;
		bool PREVIOUSLY_LOWERED = true;
		bool FIRST_TRY = true;
		// trying an optimization metric for the THRESHOLD.  Say...we want 50 detections per video,
		// we will optimize until that's right.
		
		vector<Detection> detections;
		while (true)
		{
			for (auto candidate = candidate_indices.begin(); candidate != candidate_indices.end(); ++candidate)
			{
				if (svm_responses[*candidate] > THRESHOLD)
				{
					// the BOW feature stretches from the root point (*it) to alpha away.  So if alpha is 10 and it's the first response,
					// it would be keyframes 0 to 10 (or frames 0 to 50).
					int end_index = (*candidate) + ALPHA;
				
					Detection detection;
					detection.start_frame = (*candidate) * 5;
					detection.end_frame = end_index * 5;
					detection.score = svm_responses[*candidate];
					detection.video_name = "GenericVideoName.mpg"; // [TODO]

					detections.push_back(detection);
				}
			}

			unsigned int num_detections = detections.size();
			cout << num_detections << " detections" << endl;

			if (num_detections < MIN_DETECTIONS)
			{
				// maybe there aren't enough candidates.
				if (candidate_indices.size() < MIN_DETECTIONS)
				{
					break;
					//MIN_DETECTIONS = 3 * candidate_indices.size()/4;
				}
				// too few detections, lower the threshold to allow for more.
				if (FIRST_TRY || PREVIOUSLY_LOWERED)
				{
					THRESHOLD -= STEP;
					PREVIOUSLY_LOWERED = true;
					FIRST_TRY = false;
				}
				else
				{
					// we raised it last time to allow less, but not not enough.
					// shrink the step size to get a finer grain.
					STEP -= 0.005;
					THRESHOLD -= STEP;
					PREVIOUSLY_LOWERED = true;
				}
				cout << "STEP: " << STEP << ", THRESHOLD: " << THRESHOLD << endl;
				detections.clear();
			}
			else if (num_detections > MAX_DETECTIONS)
			{
				// too many detections, raise threshold to allow less.
				if (FIRST_TRY || !PREVIOUSLY_LOWERED)
				{
					THRESHOLD += STEP;
					FIRST_TRY = false;
					PREVIOUSLY_LOWERED = false;
				}
				else
				{
					// we lowered it last time to allow more, but now we have too many.
					// shrink the step size grain for finer detail and raise the threshold by this new amount.
					STEP += 0.005;
					THRESHOLD += STEP;
					PREVIOUSLY_LOWERED = false;
				}
				detections.clear();
				cout << "STEP:" << STEP << ", THRESHOLD: " << THRESHOLD << endl;
			}
			else
			{
				// we are in the desired detection range.
				// now we can sort and print them.
				cout << "Accepting a threshold of " << THRESHOLD << " that permits " << num_detections << " events." << endl;
				break;
			}
		}
		// sort by likelihood
		std::sort(detections.begin(), detections.end());
		std::reverse(detections.begin(), detections.end());

		// print to file
		ofstream detection_stream(*it + ".detections");
		for (auto det = detections.begin(); det != detections.end(); ++det)
		{
			detection_stream << *it << ", " << det->start_frame << ", " << det->end_frame << ", " << det->score << endl;
		}
		detection_stream.close();
		cout << "-----------------------------------" << endl << endl;
	}
}

// For TRECVID detections
void computeSVMResponses()
{
	SVMInterface svm;
	directory_iterator end_iter;
	string model_file = SVM_PATH + "/model.svm";

	if (DISTRIBUTED)
	{
		SVM_PATH = "mosift/";
		model_file = "/home/lucia/data/model.svm";
	}

	cout << "SVM_PATH: " << SVM_PATH << endl;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 4, 4) == "test")
			{
				string test_file = SVM_PATH + "/" + filename;
				cout << "Testing SVM with file " << test_file << " with model " << model_file << endl;
				svm.testModelTRECVID(test_file,  model_file);
			}
		}
	}
}

// given a collection of videos, generate a single mofreak file per video,
// containing the descriptor data for that video.
void computeMoFREAKFiles()
{
	directory_iterator end_iter;

	cout << "Here are the videos: " << VIDEO_PATH << endl;
	cout << "MoFREAK files will go here: " << MOFREAK_PATH << endl;
	cout << "Motion bytes: " << NUM_MOTION_BYTES << endl;
    cout << "Appearance bytes: " << NUM_APPEARANCE_BYTES << endl;


//#pragma omp parallel
//{
//#pragma omp for private(mofreak)
    for (directory_iterator dir_iter(VIDEO_PATH); dir_iter != end_iter; ++dir_iter)
	{

		if (is_regular_file(dir_iter->status()))
		{
			// parse mosift files so first x characters gets us the video name.
			path current_file = dir_iter->path();
			string video_path = current_file.generic_string();
			string video_filename = current_file.filename().generic_string();

            if ((video_filename.substr(video_filename.length() - 3, 3) == "mp4")) //avi"))
			{
//#pragma omp single
                cout << "FILE: " << VIDEO_PATH << "/" << video_filename << endl;

				string video = VIDEO_PATH + "/" + video_filename;
				string mofreak_path = MOFREAK_PATH + "/" + video_filename + ".mofreak";

				ifstream f(mofreak_path);
				if (f.good()) {
					f.close();
//#pragma omp single
					cout << "MoFREAK already computed" << endl;
				}
				else {
					f.close();

                    cv::VideoCapture capture;
                    capture.open(video);

                    if (!capture.isOpened())
                    {
//#pragma omp single
                        cout << "Could not open file: " << video << endl;
                    }

                    mofreak->computeMoFREAKFromFile(video, capture, mofreak_path, true);

                    //mofreak->computeMoFREAKFromFile(video, mofreak_path, true);
				}
			}
		}
		else if (is_directory(dir_iter->status()))
		{
			// get folder name.
			string video_action = dir_iter->path().filename().generic_string();
//#pragma omp single
			cout << "action: " << video_action << endl;

			// set the mofreak object's action to that folder name.
			mofreak->setCurrentAction(video_action);

			// compute mofreak on all files on that folder.
			string action_video_path = VIDEO_PATH + "/" + video_action;
//#pragma omp single
			cout << "action video path: " << action_video_path << endl;

#pragma omp parallel
{
#pragma omp single nowait
            for (directory_iterator video_iter(action_video_path); video_iter != end_iter; ++video_iter)
			{
				if (is_regular_file(video_iter->status()))
				{
					string video_filename = video_iter->path().filename().generic_string();
                    if (video_filename.substr(video_filename.length() - 3, 3) == "mp4") //"avi")
					{
                        cout << "filename: " << video_filename << endl;
                        cout << "FILE: " << action_video_path << video_filename << endl;
						string mofreak_path = MOFREAK_PATH + "/" + video_action + "/" + video_filename + ".mofreak";

						// create the corresponding directories, then go ahead and compute the mofreak files.
						boost::filesystem::path dir_to_create(MOFREAK_PATH + "/" + video_action + "/");
						boost::system::error_code returned_error;
						boost::filesystem::create_directories(dir_to_create, returned_error);
						if (returned_error)
						{
							std::cout << "Could not make directory " << dir_to_create.string() << std::endl;
							exit(1);
						}

                            cout << "mofreak path: " << mofreak_path << endl;
                            //printf("num_threads = %d\n", omp_get_num_threads());
						
						ifstream f(mofreak_path);
						if (f.good()) {
							f.close();
							cout << "MoFREAK already computed" << endl;
						}	
						else {
							f.close();

                            cv::VideoCapture capture;
                            capture.open(action_video_path + "/" + video_filename);

                            if (!capture.isOpened())
                                cout << "Could not open file: " << action_video_path << "/" << video_filename << endl;

//#pragma omp task private(action_video_path, video_filename, capture)
//{
                            //printf("Thread: %d\n", omp_get_thread_num());
                            mofreak->computeMoFREAKFromFile(action_video_path + "/" + video_filename, capture, mofreak_path, true);
//}

                            //mofreak->computeMoFREAKFromFile(action_video_path + "/" + video_filename, mofreak_path, true);
						}
					}
				}
			}
        }
    }
}
}

int main(int argc, char* argv[])
{
    setParameters(); //setea NUM_CLUSTERS
    if (argc > 1)
        NUMBER_OF_GROUPS = (unsigned int) atoi(argv[1]);
    if (argc > 2)
        NUM_CLUSTERS = (unsigned int) atoi(argv[2]);
    cout << "Groups: " << NUMBER_OF_GROUPS << endl;
    cout << "Clusters: " << NUM_CLUSTERS << endl;
	clock_t start, end;
    time_t startI, endI;
	mofreak = new MoFREAKUtilities(dataset);

	if (state == DETECT_MOFREAK)
	{
		start = clock();
		computeMoFREAKFiles();
		end = clock();
	}

	// This is the most commonly used scenario.
	// Compute MoFREAK descriptors across the dataset,
	// cluster them,
	// compute the bag-of-words representation,
	// and classify.
	else if (state == DETECTION_TO_CLASSIFICATION)
	{
		start = clock();
        startI = time(NULL);
        //computeMoFREAKFiles();
        endI = time(NULL);
        cout << "COMPUTE MOFREAK FILES: " << (endI - startI)/(double)60 << " minutos! " << endl;

        if (dataset == RUGBY || dataset == KTH)
		{	
			cout << "cluster()" << endl;
            startI = time(NULL);
            //cluster();
            endI = time(NULL);
            cout << "CLUSTER FILES: " << (endI - startI)/(double)60 << " minutos! " << endl;
            cout << "computeBOWRepresentation()" << endl;
            startI = time(NULL);
            computeBOWRepresentation();
            endI = time(NULL);
            cout << "COMPUTE BOW REPRESENTATION: " << (endI - startI)/(double)60 << " minutos! " << endl;
            startI = time(NULL);
			double avg_acc = classify();
            endI = time(NULL);
            cout << "CLASSIFY: " << (endI - startI)/(double)60 << " minutos! " << endl;
		}

		//cout << "deleting mofreak..." << endl;
		//delete mofreak;
		//cout << "deleted" << endl;
		end = clock();
	}
	// TRECVID cases
	else if (state == PICK_CLUSTERS)
	{
		start = clock();
    //	pickClusters();
		end = clock();
	}
	else if (state == COMPUTE_BOW_HISTOGRAMS)
	{
		start = clock();
		const bool POSITIVE_EXAMPLES = false;
		computeBOWHistograms(POSITIVE_EXAMPLES);
		end = clock();
	}
	else if (state == DETECT)
	{
		start = clock();
		detectEvents();
		end = clock();
	}

	else if (state == TRAIN)
	{
		start = clock();
        //trainTRECVID();
		end = clock();
	}

	else if (state == GET_SVM_RESPONSES)
    {	{
            start = clock();
            computeSVMResponses();
            end = clock();
        }
		start = clock();
		computeSVMResponses();
		end = clock();
	}

    cout << "Took this long: " << (end - start)/((double)CLOCKS_PER_SEC * 60) << " seconds! " << endl;
	cout << "All done.  Press any key to continue..." << endl;
	cout << "Dumping memory leak info" << endl;
}
