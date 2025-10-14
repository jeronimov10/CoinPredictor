import psycopg2
from config_DB import config


def connect():

    try:
        connection = None
        params = config()
        print('Conectando a la base de datos de postgreSQL...')
        connection = psycopg2.connect(**params) #extraer todo de los params

        crsr = connection.cursor()
        print('PostgreSQL version')
        crsr.execute('SELECT version()')
        db_version = crsr.fetchone()
        print(db_version)
        crsr.close()


    except(Exception, psycopg2.DatabaseError) as error:
        print(error)

    finally:
        if connection is not None:
            connection.close()
            print('Conexion terminada')


connect()
