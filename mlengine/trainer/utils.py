import os
import gzip
import shutil
from six.moves import urllib
from tensorflow.python.platform import gfile


def maybe_download_and_ungzip(filename, work_directory, source_url):
    if filename[-3:] == ".gz":
        unzipped_filename = filename[:-3]
    else:
        unzipped_filename = filename

    if not gfile.Exists(work_directory):
        gfile.MakeDirs(work_directory)

    filepath = os.path.join(work_directory, filename)
    unzipped_filepath = os.path.join(work_directory, unzipped_filename)

    if not gfile.Exists(unzipped_filepath):
        urllib.request.urlretrieve(source_url, filepath)

        if not filename == unzipped_filename:
            with gzip.open(filepath, 'rb') as f_in:
                with open(unzipped_filepath, 'wb') as f_out: # remove .gz
                    shutil.copyfileobj(f_in, f_out)

        with gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded and unzipped', filename, size, 'bytes.')
    return unzipped_filepath