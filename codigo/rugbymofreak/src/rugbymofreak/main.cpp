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

#include <stdlib.h>
#include <iomanip>

#include <omp.h>

using namespace std;
using namespace boost::filesystem;

string DATA_DIR_RUGBY = "/home/lucia/Documentos/data/";
string DATA_DIR_KTH = "/home/lucia/Documentos/data/";

bool DISTRIBUTED = false;

string MOSIFT_DIR, MOFREAK_PATH, VIDEO_PATH, SVM_PATH, METADATA_PATH; //se setean en setParameters()

unsigned int NUM_MOTION_BYTES = 8;
unsigned int NUM_APPEARANCE_BYTES = 8;
unsigned int FEATURE_DIMENSIONALITY = NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES;
unsigned int NUM_CLUSTERS, NUMBER_OF_GROUPS, NUM_CLASSES;
unsigned int MAX_FEATURES_PER_FILE = 100; //cota de features a leer por cada mofreak file

vector<int> possible_classes;
deque<MoFREAKFeature> mofreak_ftrs;

enum states {DETECT_MOFREAK, DETECTION_TO_CLASSIFICATION, // standard recognition states
    PICK_CLUSTERS, COMPUTE_BOW_HISTOGRAMS, DETECT, TRAIN, GET_SVM_RESPONSES,}; // these states are exclusive to TRECVID

enum datasets {RUGBY, KTH};

int dataset = RUGBY; //KTH;
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


//setea los parametros necesarios para utilizar la herramiente de acuerdo al dataset con que se esté trabajando
//NUM_CLUSTERS, NUM_GROUPS puede reescribirse pasándoselos desde línea de consola a main
void setParameters()
{
    //RUGBY
    if (dataset == RUGBY)
    {
        NUM_CLUSTERS = 1000; //probar
        NUM_CLASSES = 3; //line, scrum, juego
        NUMBER_OF_GROUPS = 8; //no tenemos groups, lo usamos para la cantidad de partidos

        for (unsigned i = 0; i < NUM_CLASSES; ++i)
        {
            possible_classes.push_back(i);
        }

        //data a procesar y directorios para guardar resultados
        MOSIFT_DIR = DATA_DIR_RUGBY + "mosift/";
        MOFREAK_PATH = DATA_DIR_RUGBY + "mofreak/";
        VIDEO_PATH = DATA_DIR_RUGBY + "videos/";
        SVM_PATH = DATA_DIR_RUGBY + "svm/";
        METADATA_PATH = "";
    }

    //KTH
	else if (dataset == KTH)
	{
        NUM_CLUSTERS = 10;
		NUM_CLASSES = 6;
		NUMBER_OF_GROUPS = 25;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

        //data a procesar y directorios para guardar resultados
        MOSIFT_DIR = DATA_DIR_KTH + "mosift/";
        MOFREAK_PATH = DATA_DIR_KTH + "mofreak/";
        VIDEO_PATH = DATA_DIR_KTH + "videos/";
        SVM_PATH = DATA_DIR_KTH + "svm/";
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

    //por cada grupo (RUGBY -> grupo = match)
    for (int group = 1; group <= NUMBER_OF_GROUPS; group++)
    {
        std::cout << "Gathering MoFREAK Features from match " << group << std::endl;
        clustering.center_row = 0;
        //por cada clase
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
                        std::string path_group = path.substr(path.find_last_of('_') - 4, 4);
                        if (group != atoi(path_group.c_str()))
                            file_count++;
                    }
                }

                std::cout << "Number of mofreak files: " << file_count << std::endl;

                // maximum number of features to read from each file,
                // to avoid reading in too many mofreak features.
                unsigned int features_per_file = MAX_FEATURES_PER_FILE; //original 50000/file_count

                for (directory_iterator mofreak_iter(action_mofreak_path);
                    mofreak_iter != end_iter; ++mofreak_iter)
                {
                    // load each mofreak file's datamofreak_iter
                    if (is_regular_file(mofreak_iter->status()))
                    {
                        std::string path = mofreak_iter->path().string();
                        std::string path_group = path.substr(path.find_last_of('_') - 4, 4);
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


//genera un archivo mofreak por cada video que se encuentra en VIDEO_PATH
//los archivos mofreak contienen la información de los descriptores
void computeMoFREAKFiles()
{
	directory_iterator end_iter;

	cout << "Here are the videos: " << VIDEO_PATH << endl;
	cout << "MoFREAK files will go here: " << MOFREAK_PATH << endl;
	cout << "Motion bytes: " << NUM_MOTION_BYTES << endl;
    cout << "Appearance bytes: " << NUM_APPEARANCE_BYTES << endl;

    for (directory_iterator dir_iter(VIDEO_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
            path current_file = dir_iter->path();
			string video_path = current_file.generic_string();
			string video_filename = current_file.filename().generic_string();

            //checkea que el video sea .mp4
            if ((video_filename.substr(video_filename.length() - 3, 3) == "mp4")) //avi"))
			{
                string video = VIDEO_PATH + "/" + video_filename;
                cout << "FILE: " << video << endl;
				string mofreak_path = MOFREAK_PATH + "/" + video_filename + ".mofreak";

				ifstream f(mofreak_path);
				if (f.good()) {
					f.close();
					cout << "MoFREAK already computed" << endl;
				}
				else {
					f.close();

                    cv::VideoCapture capture;
                    capture.open(video);

                    if (!capture.isOpened())
                    {
                        cout << "Could not open file: " << video << endl;
                    }

                    mofreak->computeMoFREAKFromFile(video, capture, mofreak_path, true);
                    //mofreak->computeMoFREAKFromFile(video, mofreak_path, true);
				}
			}
		}
		else if (is_directory(dir_iter->status()))
		{
            //lee el nombre de la carpeta y lo usa para mostrar que tipo de acción es
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;

            //setea la acción para mofreak
			mofreak->setCurrentAction(video_action);

            string action_video_path = VIDEO_PATH + video_action;
            cout << "action video path: " << action_video_path << endl;

            //genera directorio dentro de MOFREAK_PATH para mantener separadas las distintas acciones
            boost::filesystem::path dir_to_create(MOFREAK_PATH + "/" + video_action + "/");
            boost::system::error_code returned_error;
            boost::filesystem::create_directories(dir_to_create, returned_error);
            if (returned_error)
            {
                std::cout << "Could not make directory " << dir_to_create.string() << std::endl;
                exit(1);
            }

            //calcula mofreak para todos los videos que se encuentran en ese directorio
            for (directory_iterator video_iter(action_video_path); video_iter != end_iter; ++video_iter)
			{
				if (is_regular_file(video_iter->status()))
				{
					string video_filename = video_iter->path().filename().generic_string();
                    if (video_filename.substr(video_filename.length() - 3, 3) == "mp4") //"avi")
					{
                        cout << "FILE: " << action_video_path << "/" << video_filename << endl;
						string mofreak_path = MOFREAK_PATH + "/" + video_action + "/" + video_filename + ".mofreak";

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
                            //printf("Thread: %d\n", omp_get_thread_num());

                            mofreak->computeMoFREAKFromFile(action_video_path + "/" + video_filename, capture, mofreak_path, true);
                            //mofreak->computeMoFREAKFromFile(action_video_path + "/" + video_filename, mofreak_path, true);
						}
					}
				}
			}
        }
    }
}

int main(int argc, char* argv[])
{
    setParameters();
    //NUMBER_OF_GROUPS y NUM_CLUSTERS pueden sobreescribirse pasándolos por línea de comando
    if (argc > 1)
        NUMBER_OF_GROUPS = (unsigned int) atoi(argv[1]);
    if (argc > 2)
        NUM_CLUSTERS = (unsigned int) atoi(argv[2]);
    cout << "Dataset: " << (dataset == RUGBY? "RUGBY" : "KTH") << endl;
    cout << "Classes: " << NUM_CLASSES << endl;
    cout << "Groups: " << NUMBER_OF_GROUPS << endl;
    cout << "Clusters: " << NUM_CLUSTERS << endl;
	clock_t start, end;
    time_t startI, endI;
	mofreak = new MoFREAKUtilities(dataset);

    #pragma omp parallel
    {
      int ID = omp_get_thread_num();
      cout << "hello(" << ID << ")" << endl;
      cout << "world(" << ID << ")" << endl;
    }

    //solamente genera los archivos MoFREAK
	if (state == DETECT_MOFREAK)
	{
		start = clock();
		computeMoFREAKFiles();
		end = clock();
	}

    //genera los archivos MoFREAK
    //arma los clusters y la representación del Bag-of-Words
    //clasifica usando SVM
	else if (state == DETECTION_TO_CLASSIFICATION)
	{
		start = clock();
        startI = time(NULL);
        computeMoFREAKFiles();
        endI = time(NULL);
        cout << "COMPUTE MOFREAK FILES: " << (endI - startI)/(double)60 << " minutos! " << endl;

        if (dataset == RUGBY || dataset == KTH)
		{	
			cout << "cluster()" << endl;
            startI = time(NULL);
            cluster();
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

    cout << "Tiempo total: " << (end - start)/((double)CLOCKS_PER_SEC * 60) << " minutos" << endl;
    cout << "Completo" << endl;
}
