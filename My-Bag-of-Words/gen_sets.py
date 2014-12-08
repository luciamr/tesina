from os.path import exists, isdir, basename, join, splitext
from glob import glob
import random

DATASETPATH = '/home/lucia/Documents/curso_python/python-internal-seminar/clase 2/pictures/101_ObjectCategories'
EXTENSIONS = ['.jpg', '.bmp', '.png', '.pgm', '.tif', '.tiff']

def get_categories(datasetpath):
    cat_paths = [files for files in glob(datasetpath + '/*') if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats

def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname)) for fname in glob(path + '/*') if splitext(fname)[-1].lower() in EXTENSIONS])
    all_files.sort()
    return all_files

def gen_tt_sets(n, datasetpath):
    all_files = {}
    #obtener categorias
    all_cats = get_categories(datasetpath)
    #obtener imagenes por categoria
    for cat in all_cats:
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        all_files[cat] = cat_files
    #particionar imagenes
    with open('sets.txt', 'wb') as f:
	f.write(str(len(all_cats)))
        f.write(' ')
        f.write(str(n))
        f.write('\n')
        for cat in all_cats:
            f.write(cat)
            f.write('\n')
            f.write(join(DATASETPATH, cat))
            f.write('\n')
            files = [basename(file_name) for file_name in all_files[cat]]
            for i in range(0,n):
                random.shuffle(files)
                index = int(len(files) * .3) 
                train = files[:index]
                test = files[index:]
                f.write(' '.join(train))
                f.write('\n')
                f.write(' '.join(test))
                f.write('\n')

      





