import sys
import argparse
import os
import numpy as np
import sqlite3

from timeit import default_timer

IS_PYTHON3 = sys.version_info[0] >= 3

CREATE_LOCAL_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS local_descriptors (
    image_id INT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    rows INT NOT NULL,
    cols INT NOT NULL,
    data float(256))"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    rows INT NOT NULL,
    cols INT NOT NULL,
    pixel float(2),
    FOREIGN KEY(image_id) REFERENCES local_descriptors(image_id) ON DELETE CASCADE)"""

CREATE_GLOBAL_DESCRIPTOR_TABLE = """CREATE TABLE IF NOT EXISTS global_descriptor (
    image_id INT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    data float(4096),
    FOREIGN KEY(image_id) REFERENCES local_descriptors(image_id) ON DELETE CASCADE)"""

CREATE_ALL = "; ".join([
    CREATE_LOCAL_DESCRIPTORS_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_GLOBAL_DESCRIPTOR_TABLE
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--database_dir', required=True)

    # This argument lets us only look at the matches from a certain folder.
    # We want to avoid adding matches from other folders, e.g. query. This
    # filters images according to the prefix as stored in the db file.
    # parser.add_argument('--image_prefix', required=True)

    args = parser.parse_args()
    return args


class TESTDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=TESTDatabase)

    def __init__(self, *args, **kwargs):
        super(TESTDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_local_descriptors_table = lambda: self.executescript(CREATE_LOCAL_DESCRIPTORS_TABLE)
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_global_descriptor_table = lambda: self.executescript(CREATE_GLOBAL_DESCRIPTOR_TABLE)

    def add_local_descriptors(self, image_id, name, rows, cols, data):
        self.execute(
            "INSERT INTO local_descriptors VALUES (?, ?, ?, ?, ?)",
            (image_id, name, rows, cols, data))

    def add_keypoints(self, image_id, name, rows, cols, pixel):
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?, ?)",
            (image_id, name, rows, cols, pixel))

    def add_global_descriptor(self, image_id, name, data):
        self.execute(
            "INSERT INTO global_descriptor VALUES (?, ?, ?)",
            (image_id, name, data))


def main():
    args = parse_args()
    db_list = os.listdir(args.image_dir)
    db_list.sort()
    print(db_list)
    dbimage_prenames = [image_fullname[:-4] for image_fullname in db_list]

    tini = default_timer()

    database_path = args.database_dir + '/test.db'
    if os.path.exists(database_path):
        os.remove(database_path)

    db = TESTDatabase.connect(database_path)
    db.create_tables()

    tloop = default_timer()

    for i, dbimage_prename in enumerate(dbimage_prenames):
        img = np.load(args.npz_dir + '/' + dbimage_prename + '.npz')
        print(img['local_descriptors'].shape)

        db.add_local_descriptors(i, db_list[i], img['local_descriptors'].shape[0],
                                 img['local_descriptors'].shape[1], img['local_descriptors'])

        distKp = np.array(img['keypoints'], dtype=np.float32).reshape(-1, 2)
        db.add_keypoints(i, db_list[i], img['keypoints'].shape[0], img['keypoints'].shape[1], distKp)
        db.add_global_descriptor(i, db_list[i], img['global_descriptor'])

    print("ini in %0.3fs" % (tloop - tini))
    print("loop in %0.3fs" % (default_timer() - tloop))
    db.commit()
    db.close()


if __name__ == '__main__':
    main()
