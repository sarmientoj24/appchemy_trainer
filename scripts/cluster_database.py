#### For encoding
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
    This is a function to cluster using KMeans
    the images using their encodings into n categories

    This also saves the cluster number into the database
    and keeps a database that has the cluster centroids
'''
class ClusterGenerator():
    def __init__(self, k):
        self.k = k

    def generate_cluster(self, product_df):
        kmeans = KMeans(n_clusters=self.k, max_iter=1000).fit(product_df[[
                'latent_code1', 'latent_code2', 'latent_code3',
                'latent_code4', 'latent_code5'
        ]])

        centroids = kmeans.cluster_centers_
        return kmeans.labels_, kmeans.cluster_centers_

class Apparel():
    def __init__(self):
        pass

class Apparels():
    def __init__(self):
        self.db_file = "db.sqlite3"
        self.products_df = None

    def connect_to_db(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
        except Exception as e:
            print(str(e))
        return conn

    def get_products_from_db(self):
        print("Getting products from db...")
        conn = self.connect_to_db()

        sql = '''
                SELECT id, latent_code1, latent_code2, latent_code3,
                latent_code4, latent_code5 
                FROM appchemy_apparel_dummy
            '''
        self.products_df = pd.read_sql_query(sql, conn)
        conn.close()

    def get_products_df(self):
        return self.products_df

    def set_products_df(self, products_df):
        self.products_df = products_df

    def insert_clusters_to_df(self, cluster_labels):
        self.products_df['cluster_id'] = cluster_labels

    def df_to_csv(self, filename):
        self.products_df.to_csv(filename)

    def insert_cluster_label_to_db(self, cluster_labels):
        print("Inserting Cluster Labels...")
        conn = self.connect_to_db()
        cur = conn.cursor()

        product_ids = self.products_df['id'].to_list()

        for index, cluster_label in enumerate(cluster_labels):
            sql = '''
                    UPDATE appchemy_apparel_dummy SET cluster_id = {} WHERE id = {}
                    '''.format(int(cluster_label), product_ids[index])
            cur.execute(sql)
        conn.commit()
        cur.close()

    def insert_cluster_centroids(self, cluser_centroids, create_table=False):
        print("Inserting cluster centroids...")
        conn = self.connect_to_db()
        cur = conn.cursor()

        if create_table:
            sql = "DROP TABLE IF EXISTS appchemy"
            cur.execute(sql)
            conn.commit()

            sql = '''
                    CREATE table cluster_centroids (
                        id INTEGER,
                        name TEXT,
                        latent_code1 REAL,
                        latent_code2 REAL,
                        latent_code3 REAL,
                        latent_code4 REAL,
                        latent_code5 REAL
                    ) 
                    '''
            cur.execute(sql)
            conn.commit()

        for index, cluster_centroid in enumerate(cluster_centroids):
            data_ = cluster_centroid.tolist()
            sql = '''
                    INSERT INTO appchemy_cluster_test (latent_code1, latent_code2, latent_code3,
                    latent_code4, latent_code5, cluster_id) VALUES (?, ?, ?, ?, ?, ?)
                '''.format(index)
            cur.execute(sql, (float(data_[0]), data_[1], data_[2],
                            data_[3], data_[4], index))
        conn.commit()
        conn.close()

################# UTIL CLASS #####################
class DBUtil():
    def create_connection(self, db_file):
        """ create a database connection to the SQLite database
                specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        conn = None
        try:
                conn = sqlite3.connect(db_file)
        except Error as e:
                print(e)
        return conn

    def drop_tbl(self, table_name):
        database = "db.sqlite3"
        conn = self.create_connection(database)

        sql = "DROP TABLE {}".format(table_name)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        conn.close()

    def create_encoded_table(self):
        database = "db.sqlite3"
        conn = self.create_connection(database)

        sql = '''
            CREATE TABLE product_entries (
                    id INTEGER,
                    name TEXT,
                    old_name TEXT,
                    cluster_id INTEGER,
                    subcategory TEXT,
                    z_mean_1 REAL, z_log_var_1 REAL, z_fixed_code_1 REAL,
                    z_mean_2 REAL, z_log_var_2 REAL, z_fixed_code_2 REAL,
                    z_mean_3 REAL, z_log_var_3 REAL, z_fixed_code_3 REAL, 
                    z_mean_4 REAL, z_log_var_4 REAL, z_fixed_code_4 REAL,
                    z_mean_5 REAL, z_log_var_5 REAL, z_fixed_code_5 REAL,
                    z_mean_6 REAL, z_log_var_6 REAL, z_fixed_code_6 REAL,
                    z_mean_7 REAL, z_log_var_7 REAL, z_fixed_code_7 REAL,
                    z_mean_8 REAL, z_log_var_8 REAL, z_fixed_code_8 REAL,
                    z_mean_9 REAL, z_log_var_9 REAL, z_fixed_code_9 REAL,
                    z_mean_10 REAL, z_log_var_10 REAL, z_fixed_code_10 REAL,
                    z_mean_11 REAL, z_log_var_11 REAL, z_fixed_code_11 REAL,
                    z_mean_12 REAL, z_log_var_12 REAL, z_fixed_code_12 REAL,
                    z_mean_13 REAL, z_log_var_13 REAL, z_fixed_code_13 REAL,
                    z_mean_14 REAL, z_log_var_14 REAL, z_fixed_code_14 REAL,
                    z_mean_15 REAL, z_log_var_15 REAL, z_fixed_code_15 REAL,
                    z_mean_16 REAL, z_log_var_16 REAL, z_fixed_code_16 REAL
                ) 
            '''
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        conn.close()

    def cluster_table(self):
        database = "db.sqlite3"
        conn = self.create_connection(database)

        sql = '''
            CREATE TABLE cluster_centroids (
                id INTEGER,
                cluster_id INTEGER,
                xtra_field TEXT,
                z_mean_1 REAL, z_log_var_1 REAL, z_fixed_code_1 REAL,
                z_mean_2 REAL, z_log_var_2 REAL, z_fixed_code_2 REAL,
                z_mean_3 REAL, z_log_var_3 REAL, z_fixed_code_3 REAL, 
                z_mean_4 REAL, z_log_var_4 REAL, z_fixed_code_4 REAL,
                z_mean_5 REAL, z_log_var_5 REAL, z_fixed_code_5 REAL,
                z_mean_6 REAL, z_log_var_6 REAL, z_fixed_code_6 REAL,
                z_mean_7 REAL, z_log_var_7 REAL, z_fixed_code_7 REAL,
                z_mean_8 REAL, z_log_var_8 REAL, z_fixed_code_8 REAL,
                z_mean_9 REAL, z_log_var_9 REAL, z_fixed_code_9 REAL,
                z_mean_10 REAL, z_log_var_10 REAL, z_fixed_code_10 REAL,
                z_mean_11 REAL, z_log_var_11 REAL, z_fixed_code_11 REAL,
                z_mean_12 REAL, z_log_var_12 REAL, z_fixed_code_12 REAL,
                z_mean_13 REAL, z_log_var_13 REAL, z_fixed_code_13 REAL,
                z_mean_14 REAL, z_log_var_14 REAL, z_fixed_code_14 REAL,
                z_mean_15 REAL, z_log_var_15 REAL, z_fixed_code_15 REAL,
                z_mean_16 REAL, z_log_var_16 REAL, z_fixed_code_16 REAL
            ) 
        '''

    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    conn.close()

    def insert_dummy_data_to_sqlite(self):
        database = "db.sqlite3"
        conn = self.create_connection(database)

        sql = '''
                SELECT id
                FROM appchemy_apparel_dummy
            '''
        products_df = pd.read_sql_query(sql, conn)
        ids = products_df['id'].to_list()

        latent_code1 = np.random.uniform(-2, 2, size=(len(ids),))
        latent_code2 = np.random.uniform(-2, 2, size=(len(ids),))
        latent_code3 = np.random.uniform(-2, 2, size=(len(ids),))
        latent_code4 = np.random.uniform(-2, 2, size=(len(ids),))
        latent_code5 = np.random.uniform(-2, 2, size=(len(ids),))
        
        cur = conn.cursor()
        
        for index in range(len(ids)):
            sql = '''
                UPDATE appchemy_apparel_dummy SET latent_code1 = {}, 
                latent_code2 = {}, latent_code3 = {},
                latent_code4 = {}, latent_code5 = {} WHERE id = {}
            '''.format(latent_code1[index], latent_code2[index], 
                latent_code3[index], latent_code4[index], latent_code5[index],
                ids[index])
            cur.execute(sql)
            if index % 200 == 0:
                conn.commit()
        conn.commit()
        conn.close()


################ PROGRAM START ###################
if __name__ == "__main__":
    # prod = Apparels()
    # cluster_gen = ClusterGenerator(k=5)
    # prod.get_products_from_db()