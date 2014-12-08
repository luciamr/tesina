import libsvm
import argparse
from cPickle import load
from learn import extractSift, computeHistograms, writeHistogramsToFile
from load_sets import read_tt_sets_file

HISTOGRAMS_FILE = 'testdata.svm'
CODEBOOK_FILE = 'codebook.file'
MODEL_FILE = 'trainingdata.svm.model'
DATASETPATH = '101_ObjectCategories'
NUM_SETS = 5

def parse_arguments():
    parser = argparse.ArgumentParser(description='classify images with a visual bag of words model')
    parser.add_argument('-c', help='path to the codebook file', required=False, default=CODEBOOK_FILE)
    parser.add_argument('-m', help='path to the model  file', required=False, default=MODEL_FILE)
    parser.add_argument('input_images', help='images to classify', nargs='+')
    args = parser.parse_args()
    return args

cats, files = read_tt_sets_file()

for set in range(0, NUM_SETS):
    print "---------------------"
    print "## test set " + str(set)
    
    print "---------------------"
    print "## extract Sift features"
    all_files = []
    all_files_labels = {}
    all_features = {}
    
    for cat_path in files.keys():
        cat_files = [cat_path + '/' + name for name in files[cat_path][set][1][0]]
    all_files.extend(cat_files)        

    datasetpath = DATASETPATH
    model_file = datasetpath + str(set) + '.' + MODEL_FILE
    codebook_file = datasetpath + str(set) + '.' + CODEBOOK_FILE
    fnames = all_files
    all_features = extractSift(fnames)

    for i in fnames:
        all_files_labels[i] = 0  # label is unknown

    print "---------------------"
    print "## loading codebook from " + codebook_file
    with open(codebook_file, 'rb') as f:
        codebook = load(f)

    print "---------------------"
    print "## computing visual word histograms"
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print "---------------------"
    print "## write the histograms to file to pass it to the svm"
    print all_word_histgrams.keys()
    nclusters = codebook.shape[0]
    writeHistogramsToFile(nclusters,
                          all_files_labels,
                          fnames,
                          all_word_histgrams,
                          datasetpath + str(set) + '.' + HISTOGRAMS_FILE)

    print "---------------------"
    print "## test data with svm"
    print libsvm.test(datasetpath + str(set) + '.' + HISTOGRAMS_FILE, model_file)
